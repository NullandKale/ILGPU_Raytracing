// ==============================
// Engine/Framebuffer.cs (zero-copy ready; external color view support) + ReSTIR reservoirs (SoA)
// ==============================
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using ILGPU;
using System;
using ILGPU_Raytracing.Engine;
using ILGPU.Algorithms;
using System.Runtime.CompilerServices;

public sealed class Framebuffer : IDisposable
{
    private const int kGpuBuffers = 3;
    private readonly CudaAccelerator _cuda;
    private readonly CudaStream _stream;

    // NOTE: color[] is still allocated for compatibility, but can be overridden with an external view
    private MemoryBuffer1D<int, Stride1D.Dense>[] _color = new MemoryBuffer1D<int, Stride1D.Dense>[kGpuBuffers];
    private MemoryBuffer1D<float, Stride1D.Dense>[] _depth = new MemoryBuffer1D<float, Stride1D.Dense>[kGpuBuffers];
    private MemoryBuffer1D<int, Stride1D.Dense>[] _objectId = new MemoryBuffer1D<int, Stride1D.Dense>[kGpuBuffers];
    private MemoryBuffer1D<int, Stride1D.Dense>[] _cameraId = new MemoryBuffer1D<int, Stride1D.Dense>[kGpuBuffers];

    // ---------- ReSTIR ping-pong reservoirs as SoA per slot ----------
    private struct ResSoA
    {
        public MemoryBuffer1D<Float3, Stride1D.Dense> L;
        public MemoryBuffer1D<Float3, Stride1D.Dense> wi;
        public MemoryBuffer1D<float, Stride1D.Dense> pdf;
        public MemoryBuffer1D<float, Stride1D.Dense> w;
        public MemoryBuffer1D<float, Stride1D.Dense> wSum;
        public MemoryBuffer1D<int, Stride1D.Dense> m;
        public MemoryBuffer1D<int, Stride1D.Dense> lightId;

        public void Dispose()
        {
            L?.Dispose(); wi?.Dispose(); pdf?.Dispose(); w?.Dispose();
            wSum?.Dispose(); m?.Dispose(); lightId?.Dispose();
            L = null; wi = null; pdf = null; w = null; wSum = null; m = null; lightId = null;
        }
    }

    private ResSoA[] _resA = new ResSoA[kGpuBuffers];
    private ResSoA[] _resB = new ResSoA[kGpuBuffers];

    private int[][] _cpuColor = new int[kGpuBuffers][];
    private float[][] _cpuDepth = new float[kGpuBuffers][];
    private int[][] _cpuObjectId = new int[kGpuBuffers][];

    private int _length = 0;

    public Framebuffer(CudaAccelerator cuda, CudaStream stream)
    {
        _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda));
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    public int Length => _length;

    public void EnsureLength(int length)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (length == _length && _color[0] != null) return;

        DisposeBuffers();
        _length = length;

        for (int i = 0; i < kGpuBuffers; i++)
        {
            _color[i] = _cuda.Allocate1D<int>(length);
            _depth[i] = _cuda.Allocate1D<float>(length);
            _objectId[i] = _cuda.Allocate1D<int>(length);
            _cameraId[i] = _cuda.Allocate1D<int>(1);

            // Allocate SoA reservoirs for A/B
            _resA[i] = AllocateReservoirSoA(length);
            _resB[i] = AllocateReservoirSoA(length);

            _cpuColor[i] = new int[length];
            _cpuDepth[i] = new float[length];
            _cpuObjectId[i] = new int[length];
        }
    }

    private ResSoA AllocateReservoirSoA(int length)
    {
        return new ResSoA
        {
            L = _cuda.Allocate1D<Float3>(length),
            wi = _cuda.Allocate1D<Float3>(length),
            pdf = _cuda.Allocate1D<float>(length),
            w = _cuda.Allocate1D<float>(length),
            wSum = _cuda.Allocate1D<float>(length),
            m = _cuda.Allocate1D<int>(length),
            lightId = _cuda.Allocate1D<int>(length),
        };
    }

    public GpuFramebuffer GetGpu(int slot)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (_color[slot] == null) throw new InvalidOperationException("Framebuffer not allocated. Call EnsureLength() first.");
        GpuFramebuffer fb = default;
        fb.color = _color[slot].View;
        fb.depth = _depth[slot].View;
        fb.objectId = _objectId[slot].View;
        fb.cameraId = _cameraId[slot].View;
        return fb;
    }

    // Same as GetGpu() but overrides the color target with an externally provided CUDA view (e.g., mapped GL PBO)
    public GpuFramebuffer GetGpuWithExternalColor(int slot, in ArrayView<int> externalColor)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (_depth[slot] == null) throw new InvalidOperationException("Framebuffer not allocated. Call EnsureLength() first.");
        if (externalColor.Length != _length) throw new ArgumentException("External color view length must match framebuffer length.", nameof(externalColor));

        GpuFramebuffer fb = default;
        fb.color = externalColor;
        fb.depth = _depth[slot].View;
        fb.objectId = _objectId[slot].View;
        fb.cameraId = _cameraId[slot].View;
        return fb;
    }

    // ReSTIR (SoA): get prev/current reservoir views for a given slot and frame index
    public void GetReservoirPair(int slot, int frame, out GpuReservoirSoA prev, out GpuReservoirSoA cur)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (_resA[slot].L == null || _resB[slot].L == null) throw new InvalidOperationException("Framebuffer not allocated. Call EnsureLength() first.");

        bool even = (frame & 1) == 0;
        var a = _resA[slot];
        var b = _resB[slot];

        if (even)
        {
            prev = new GpuReservoirSoA { L = b.L.View, wi = b.wi.View, pdf = b.pdf.View, w = b.w.View, wSum = b.wSum.View, m = b.m.View, lightId = b.lightId.View };
            cur = new GpuReservoirSoA { L = a.L.View, wi = a.wi.View, pdf = a.pdf.View, w = a.w.View, wSum = a.wSum.View, m = a.m.View, lightId = a.lightId.View };
        }
        else
        {
            prev = new GpuReservoirSoA { L = a.L.View, wi = a.wi.View, pdf = a.pdf.View, w = a.w.View, wSum = a.wSum.View, m = a.m.View, lightId = a.lightId.View };
            cur = new GpuReservoirSoA { L = b.L.View, wi = b.wi.View, pdf = b.pdf.View, w = b.w.View, wSum = b.wSum.View, m = b.m.View, lightId = b.lightId.View };
        }
    }

    public void DownloadToCpu(int slot)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (_color[slot] == null) throw new InvalidOperationException("Framebuffer not allocated.");
        _color[slot].CopyToCPU(_cpuColor[slot]);
        _depth[slot].CopyToCPU(_cpuDepth[slot]);
        _objectId[slot].CopyToCPU(_cpuObjectId[slot]);
        _stream.Synchronize();
    }

    public int[] CpuColor(int slot) { if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot)); return _cpuColor[slot]; }
    public float[] CpuDepth(int slot) { if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot)); return _cpuDepth[slot]; }
    public int[] CpuObjectId(int slot) { if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot)); return _cpuObjectId[slot]; }

    public int ReadCameraId(int slot)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (_cameraId[slot] == null) throw new InvalidOperationException("Framebuffer not allocated.");
        int[] tmp = new int[1];
        _cameraId[slot].CopyToCPU(tmp);
        _stream.Synchronize();
        return tmp[0];
    }

    // Kept for compatibility but unused in the zero-copy path
    public void CopyColorToPbo(int slot, CudaGlInteropIndexBuffer pbo)
    {
        if ((uint)slot >= kGpuBuffers) throw new ArgumentOutOfRangeException(nameof(slot));
        if (pbo == null) throw new ArgumentNullException(nameof(pbo));
        if (_color[slot] == null) throw new InvalidOperationException("Framebuffer not allocated.");

        pbo.MapCuda(_stream);
        var dst = pbo.GetCudaArrayView();
        var src = _color[slot].View;
        src.CopyTo(dst);
        _stream.Synchronize();
        pbo.UnmapCuda(_stream);
    }

    private void DisposeBuffers()
    {
        for (int i = 0; i < kGpuBuffers; i++)
        {
            _color[i]?.Dispose();
            _depth[i]?.Dispose();
            _objectId[i]?.Dispose();
            _cameraId[i]?.Dispose();
            _resA[i].Dispose();
            _resB[i].Dispose();
            _color[i] = null;
            _depth[i] = null;
            _objectId[i] = null;
            _cameraId[i] = null;
            _resA[i] = default;
            _resB[i] = default;
            _cpuColor[i] = null;
            _cpuDepth[i] = null;
            _cpuObjectId[i] = null;
        }
    }

    public void Dispose() => DisposeBuffers();
}

// Your existing GBuffer stays as an owning host-side class
public sealed class GBuffer : IDisposable
{
    private readonly CudaAccelerator _cuda;
    private readonly CudaStream _stream;
    private MemoryBuffer1D<Float3, Stride1D.Dense> _worldPos;
    private MemoryBuffer1D<Float3, Stride1D.Dense> _normalWS;
    private MemoryBuffer1D<Float3, Stride1D.Dense> _baseColor;
    private MemoryBuffer1D<int, Stride1D.Dense> _matId;
    private MemoryBuffer1D<int, Stride1D.Dense> _objId;
    private MemoryBuffer1D<int, Stride1D.Dense> _hitMask;
    private int _length;

    public GBuffer(CudaAccelerator cuda, CudaStream stream)
    { _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda)); _stream = stream ?? throw new ArgumentNullException(nameof(stream)); }

    public void EnsureLength(int length)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (_length == length && _worldPos != null) return;
        DisposeBuffers();
        _length = length;
        _worldPos = _cuda.Allocate1D<Float3>(length);
        _normalWS = _cuda.Allocate1D<Float3>(length);
        _baseColor = _cuda.Allocate1D<Float3>(length);
        _matId = _cuda.Allocate1D<int>(length);
        _objId = _cuda.Allocate1D<int>(length);
        _hitMask = _cuda.Allocate1D<int>(length);
    }

    public GpuGBuffer GetGpu()
    {
        if (_worldPos == null) throw new InvalidOperationException("GBuffer not allocated. Call EnsureLength() first.");
        GpuGBuffer gb = default;
        gb.worldPos = _worldPos.View;
        gb.normalWS = _normalWS.View;
        gb.baseColor = _baseColor.View;
        gb.matId = _matId.View;
        gb.objId = _objId.View;
        gb.hitMask = _hitMask.View;
        return gb;
    }

    private void DisposeBuffers()
    {
        _worldPos?.Dispose();
        _normalWS?.Dispose();
        _baseColor?.Dispose();
        _matId?.Dispose();
        _objId?.Dispose();
        _hitMask?.Dispose();
        _worldPos = null;
        _normalWS = null;
        _baseColor = null;
        _matId = null;
        _objId = null;
        _hitMask = null;
    }

    public void Dispose() => DisposeBuffers();
}
