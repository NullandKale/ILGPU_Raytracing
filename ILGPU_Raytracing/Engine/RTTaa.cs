
// ==============================
// Engine/RTTaa.cs
// ==============================
using System;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Mathematics;

namespace ILGPU_Raytracing.Engine
{
    public sealed class RTTaa : IDisposable
    {
        private readonly CudaAccelerator _cuda;
        private readonly CudaStream _stream;

        private MemoryBuffer1D<int, Stride1D.Dense> _historyColor;
        private MemoryBuffer1D<int, Stride1D.Dense> _historyObjId;
        private bool _historyValid;
        private int _histW = -1, _histH = -1;

        private readonly Action<Index1D, TaaParams> _taaKernel;

        public RTTaa(CudaAccelerator cuda, CudaStream stream)
        {
            _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda));
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _taaKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, TaaParams>(TaaResolveKernel);
        }

        public void Ensure(int outW, int outH)
        {
            int len = Math.Max(1, outW * outH);
            if (_historyColor == null || _historyColor.Length != len)
            {
                _historyColor?.Dispose();
                _historyObjId?.Dispose();
                _historyColor = _cuda.Allocate1D<int>(len);
                _historyObjId = _cuda.Allocate1D<int>(len);
                _historyValid = false;
                _histW = outW;
                _histH = outH;
            }
        }

        public void ResolveUpsample(
            ArrayView<int> outColorFullRes,
            ArrayView<int> lowColor,
            ArrayView<int> lowObjId,
            int inW, int inH,
            int outW, int outH,
            int frame,
            Camera prevCam,
            Camera curCam)
        {
            if (_historyColor == null || _historyColor.Length != outColorFullRes.Length)
            {
                Ensure(outW, outH);
            }

            TaaParams p = default;
            p.outColor = outColorFullRes;
            p.inColorLow = lowColor;
            p.inObjIdLow = lowObjId;
            p.historyColor = _historyColor.View;
            p.historyObjId = _historyObjId.View;

            p.outW = outW;
            p.outH = outH;
            p.inW = inW;
            p.inH = inH;

            // Tunables
            p.feedback = 0.075f;            // lower = more history
            p.sharpness = 0.10f;           // lightweight unsharp amount
            p.clampK = 1.25f;              // neighborhood clamp strength
            p.isFirstFrame = _historyValid ? 0 : 1;

            // No motion vectors (engine-side) — rely on disocclusion via objId + local variance
            p.motionScaleX = 0f;
            p.motionScaleY = 0f;

            int total = Math.Max(1, outW * outH);
            _taaKernel(new Index1D(total), p);

            _historyValid = true;
        }

        private struct TaaParams
        {
            public ArrayView<int> outColor;
            public ArrayView<int> inColorLow;
            public ArrayView<int> inObjIdLow;
            public ArrayView<int> historyColor;
            public ArrayView<int> historyObjId;

            public int outW;
            public int outH;
            public int inW;
            public int inH;

            public float feedback;
            public float sharpness;
            public float clampK;
            public int isFirstFrame;

            public float motionScaleX;
            public float motionScaleY;
        }

        // ---------------------------------------------
        // TAAU kernel: Catmull-Rom upsampling + history reprojection (no motion), neighborhood clamp, light sharpening
        // ---------------------------------------------
        private static void TaaResolveKernel(Index1D idx, TaaParams p)
        {
            int outW = p.outW;
            int outH = p.outH;
            if (idx >= outW * p.outH) return;

            int px = idx % outW;
            int py = idx / outW;

            // Map output pixel center to low-res continuous coords
            float sx = ((float)px + 0.5f) * ((float)p.inW / (float)outW) - 0.5f;
            float sy = ((float)py + 0.5f) * ((float)p.inH / (float)outH) - 0.5f;

            Float3 cur = SampleCatRomSRGB(p.inColorLow, p.inW, p.inH, sx, sy);

            // Simple 3x3 neighborhood min/max in current (derived from multiple CatRom taps)
            Float3 nmin = cur;
            Float3 nmax = cur;
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (ox == 0 && oy == 0) continue;
                    Float3 c = SampleCatRomSRGB(p.inColorLow, p.inW, p.inH, sx + ox * 0.5f, sy + oy * 0.5f);
                    nmin = new Float3(XMath.Min(nmin.X, c.X), XMath.Min(nmin.Y, c.Y), XMath.Min(nmin.Z, c.Z));
                    nmax = new Float3(XMath.Max(nmax.X, c.X), XMath.Max(nmax.Y, c.Y), XMath.Max(nmax.Z, c.Z));
                }
            }

            // Nearest objId from low-res for disocclusion
            int objId = SampleNearestObj(p.inObjIdLow, p.inW, p.inH, sx, sy);

            // History fetch (no motion reprojection here)
            Float3 hist = UnpackSRGB(p.historyColor[idx]);

            // Disocclusion: if obj changed, drop history
            int histObj = p.historyObjId[idx];
            bool reset = (p.isFirstFrame != 0) || (histObj != objId);

            // Clamp history into current neighborhood
            Float3 histClamped = Clamp(hist, nmin, nmax, p.clampK);

            // Temporal blend
            float a = reset ? 1.0f : p.feedback;
            Float3 accum = Lerp(histClamped, cur, a);

            // Light sharpening (unsharp mask in linear domain)
            Float3 sharpen = accum * (1.0f + 2.0f * p.sharpness) - (nmin + nmax) * (0.5f * p.sharpness);
            accum = Mix(accum, sharpen, p.sharpness);

            // Write out & update history
            p.outColor[idx] = PackSRGB(accum);
            p.historyColor[idx] = p.outColor[idx];
            p.historyObjId[idx] = objId;
        }

        // ---------------- helpers (all inline-friendly) ----------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Lerp(Float3 a, Float3 b, float t)
        {
            return a * (1f - t) + b * t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Mix(Float3 a, Float3 b, float t)
        {
            return a * (1f - t) + b * t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Clamp(Float3 v, Float3 lo, Float3 hi, float k)
        {
            Float3 cmin = new Float3(lo.X - k * 0.0f, lo.Y - k * 0.0f, lo.Z - k * 0.0f);
            Float3 cmax = new Float3(hi.X + k * 0.0f, hi.Y + k * 0.0f, hi.Z + k * 0.0f);
            return new Float3(XMath.Min(cmax.X, XMath.Max(cmin.X, v.X)),
                              XMath.Min(cmax.Y, XMath.Max(cmin.Y, v.Y)),
                              XMath.Min(cmax.Z, XMath.Max(cmin.Z, v.Z)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int SampleNearestObj(ArrayView<int> a, int w, int h, float sx, float sy)
        {
            int ix = XMath.Clamp((int)XMath.Round(sx), 0, w - 1);
            int iy = XMath.Clamp((int)XMath.Round(sy), 0, h - 1);
            return a[iy * w + ix];
        }

        // Catmull-Rom reconstruction in sRGB space with linearized blending for stability
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SampleCatRomSRGB(ArrayView<int> a, int w, int h, float x, float y)
        {
            int x1 = XMath.Clamp((int)XMath.Floor(x), 0, w - 1);
            int y1 = XMath.Clamp((int)XMath.Floor(y), 0, h - 1);
            float fx = x - x1;
            float fy = y - y1;

            Float3 c00 = UnpackSRGB(a[y1 * w + x1]);
            Float3 c10 = UnpackSRGB(a[y1 * w + XMath.Min(x1 + 1, w - 1)]);
            Float3 c01 = UnpackSRGB(a[XMath.Min(y1 + 1, h - 1) * w + x1]);
            Float3 c11 = UnpackSRGB(a[XMath.Min(y1 + 1, h - 1) * w + XMath.Min(x1 + 1, w - 1)]);

            Float3 cx0 = CatRom(c00, c10, fx);
            Float3 cx1 = CatRom(c01, c11, fx);
            return CatRom(cx0, cx1, fy);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 CatRom(Float3 a, Float3 b, float t)
        {
            // Simplified edge-preserving: interpolate between two samples (acts like smoothstep/bicubic-lite)
            float tt = t * (2f - t);
            return a * (1f - tt) + b * tt;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 UnpackSRGB(int rgba)
        {
            float r = ((rgba >> 16) & 255) / 255.0f;
            float g = ((rgba >> 8) & 255) / 255.0f;
            float b = (rgba & 255) / 255.0f;
            // Approximate sRGB->linear
            r = (r <= 0.04045f) ? (r / 12.92f) : XMath.Pow((r + 0.055f) / 1.055f, 2.4f);
            g = (g <= 0.04045f) ? (g / 12.92f) : XMath.Pow((g + 0.055f) / 1.055f, 2.4f);
            b = (b <= 0.04045f) ? (b / 12.92f) : XMath.Pow((b + 0.055f) / 1.055f, 2.4f);
            return new Float3(r, g, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PackSRGB(Float3 c)
        {
            float rL = XMath.Max(0f, XMath.Min(1f, c.X));
            float gL = XMath.Max(0f, XMath.Min(1f, c.Y));
            float bL = XMath.Max(0f, XMath.Min(1f, c.Z));
            // linear->sRGB
            float r = (rL <= 0.0031308f) ? 12.92f * rL : 1.055f * XMath.Pow(rL, 1f / 2.4f) - 0.055f;
            float g = (gL <= 0.0031308f) ? 12.92f * gL : 1.055f * XMath.Pow(gL, 1f / 2.4f) - 0.055f;
            float b = (bL <= 0.0031308f) ? 12.92f * bL : 1.055f * XMath.Pow(bL, 1f / 2.4f) - 0.055f;
            int R = (int)XMath.Round(XMath.Max(0f, XMath.Min(1f, r)) * 255f);
            int G = (int)XMath.Round(XMath.Max(0f, XMath.Min(1f, g)) * 255f);
            int B = (int)XMath.Round(XMath.Max(0f, XMath.Min(1f, b)) * 255f);
            return (255 << 24) | (R << 16) | (G << 8) | B;
        }

        public void Dispose()
        {
            _historyColor?.Dispose();
            _historyObjId?.Dispose();
        }
    }
}
