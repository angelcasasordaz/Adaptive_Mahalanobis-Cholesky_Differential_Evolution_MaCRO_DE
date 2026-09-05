"""Device-resident NumPy-compatible PCG64 plans for the two MaCRO batches.

SeedSequence initialization stays in NumPy, once per run. CUDA retains the
128-bit PCG state and the cached upper uint32 across every epoch. Sampling uses
NumPy 2.2's Floyd/32-bit Lemire/shuffle order, including rejection draws.
No fast math or fused multiply-add: uniform/scale rounding must match NumPy.
"""
import numpy as np


_SOURCE = r'''
typedef unsigned long long U;
struct RNG {
    U lo, hi, il, ih, cached, has;
    __device__ U next64() {
        U low = lo * 4865540595714422341ULL;
        U high = __umul64hi(lo, 4865540595714422341ULL)
               + hi * 4865540595714422341ULL + lo * 2549297995355413924ULL;
        lo = low + il; hi = high + ih + (lo < low);
        U x = hi ^ lo; unsigned r = hi >> 58;
        return (x >> r) | (x << ((-r) & 63));
    }
    __device__ unsigned next32() {
        if (has) { has = 0; return (unsigned)cached; }
        U x = next64(); cached = x >> 32; has = 1; return (unsigned)x;
    }
    __device__ unsigned bounded(unsigned max) {
        if (!max) return 0;
        unsigned range = max + 1;
        U product = (U)next32() * range;
        unsigned low = (unsigned)product;
        if (low < range) {
            unsigned threshold = (0xffffffffU - max) % range;
            while (low < threshold) {
                product = (U)next32() * range; low = (unsigned)product;
            }
        }
        return product >> 32;
    }
    __device__ double uniform() { return (next64() >> 11) * 0x1.0p-53; }
    __device__ void choice3(int n, int *out) {
        for (int k=0; k<3; ++k) {
            int j=n-3+k, value=bounded(j);
            for (int i=0; i<k; ++i) if (out[i]==value) { value=j; break; }
            out[k]=value;
        }
        for (int i=2; i>0; --i) {
            int j=bounded(i), t=out[j]; out[j]=out[i]; out[i]=t;
        }
    }
};
__device__ double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
// Match NumPy's contiguous float64 pairwise sum (128-element blocks).
__device__ double pairwise(const double *x, int n) {
    if (n<8) { double s=-0.0; for(int i=0;i<n;++i) s+=x[i]; return s; }
    if (n<=128) {
        double s[8]; for(int i=0;i<8;++i) s[i]=x[i];
        int i=8;
        for(;i<n-(n%8);i+=8) for(int j=0;j<8;++j) s[j]+=x[i+j];
        double total=((s[0]+s[1])+(s[2]+s[3]))+((s[4]+s[5])+(s[6]+s[7]));
        for(;i<n;++i) total+=x[i]; return total;
    }
    int half=n/2; half-=half%8;
    return pairwise(x,half)+pairwise(x+half,n-half);
}
extern "C" __global__ void fmean(const double *f, double *out, int runs, int size) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<runs) out[r]=pairwise(f+r*size,size)/size;
}
extern "C" __global__ void plan(
    U *states, const bool *close, const double *div, long long *donors,
    bool *cross, double *f, double *pcr_out, int *pools,
    int runs, int pop, int dims, int macro, double beta_min,
    double beta_max, double pcr, double cr) {
    int r=blockIdx.x * blockDim.x + threadIdx.x;
    if (r>=runs) return;
    U *s=states+r*6;
    RNG rng={s[0],s[1],s[2],s[3],s[4],s[5]};
    int *pool=pools+r*pop, count=0;
    bool choose_close=div[r]>=0.5;
    for (int i=0; i<pop; ++i)
        if (close[r*pop+i]==choose_close) pool[count++]=i;
    if (macro && count<3) { count=pop; for(int i=0;i<pop;++i) pool[i]=i; }
    double pc=macro ? clip(pcr+0.25*(1.0-div[r]),0.10,0.95) : cr;
    double scale=clip(0.5+(1.0-div[r]),0.5,1.5);
    pcr_out[r]=pc;
    for (int i=0; i<pop; ++i) {
        int excluded=-1;
        if (!macro) for(int j=0;j<count;++j) if(pool[j]==i) excluded=j;
        int size=count-(excluded>=0), selected[3];
        bool fallback=!macro && size<3;
        rng.choice3(fallback ? pop-1 : (macro ? count : size), selected);
        for(int k=0;k<3;++k) {
            int index=selected[k];
            donors[(r*pop+i)*3+k]=fallback ? index+(index>=i)
                : pool[index+(!macro && excluded>=0 && index>=excluded)];
        }
        int base=(r*pop+i)*dims;
        if (macro) for(int d=0;d<dims;++d)
            f[base+d]=clip((beta_min+(beta_max-beta_min)*rng.uniform())*scale,0.10,1.50);
        int j0=rng.bounded(dims-1);
        for(int d=0;d<dims;++d) cross[base+d]=rng.uniform()<=pc;
        cross[base+j0]=true;
    }
    s[0]=rng.lo;s[1]=rng.hi;s[2]=rng.il;s[3]=rng.ih;s[4]=rng.cached;s[5]=rng.has;
}
'''


class MacroRandomPlan:
    def __init__(self, cp, generators, pop_size, n_dims):
        self.cp = cp
        words = []
        mask = (1 << 64) - 1
        for generator in generators:
            state = generator.bit_generator.state
            if state['bit_generator'] != 'PCG64':
                raise TypeError('MaCRO device plans require the existing PCG64 stream')
            inner = state['state']
            words.append([inner['state'] & mask, inner['state'] >> 64,
                          inner['inc'] & mask, inner['inc'] >> 64,
                          state['uinteger'], state['has_uint32']])
        self.states = cp.asarray(words, dtype=cp.uint64)
        self.runs, self.pop, self.dims = len(generators), pop_size, n_dims
        self.donors = cp.empty((self.runs, pop_size, 3), dtype=cp.int64)
        self.cross = cp.empty((self.runs, pop_size, n_dims), dtype=cp.bool_)
        self.f = cp.empty(self.cross.shape, dtype=cp.float64)
        self.pcr = cp.empty(self.runs, dtype=cp.float64)
        self.pools = cp.empty((self.runs, pop_size), dtype=cp.int32)
        self.kernel = cp.RawKernel(_SOURCE, 'plan', options=('--fmad=false',))
        self.mean_kernel = cp.RawKernel(_SOURCE, 'fmean', options=('--fmad=false',))
        self.fmean = cp.empty(self.runs, dtype=cp.float64)

    def mean(self):
        self.mean_kernel(((self.runs + 31)//32,), (32,), (
            self.f, self.fmean, np.int32(self.runs), np.int32(self.pop*self.dims)))
        return self.fmean

    def generate(self, close, div, engine):
        self.kernel(((self.runs + 31)//32,), (32,), (
            self.states, close, div, self.donors, self.cross, self.f, self.pcr, self.pools,
            np.int32(self.runs), np.int32(self.pop), np.int32(self.dims),
            np.int32(engine.optimizer_name == 'MaCRO-DE'), np.float64(engine.beta_min),
            np.float64(engine.beta_max), np.float64(engine.pcr), np.float64(engine.cr)))
        return self.donors, self.cross, self.f, self.pcr

    def export_states(self, generators):
        # Only at the batch boundary, keeping callers' continuation state exact.
        for generator, words in zip(generators, self.cp.asnumpy(self.states)):
            lo, hi, il, ih, cached, has = map(int, words)
            generator.bit_generator.state = dict(bit_generator='PCG64',
                state=dict(state=(hi << 64) | lo, inc=(ih << 64) | il),
                uinteger=cached, has_uint32=has)
