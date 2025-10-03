// ==============================
// GPU-side data and kernels (struct-oriented API) + ReSTIR DI (SoA reservoirs) + multi-bounce GI
// Fixes in this revision:
// • Temporal reuse uses camera reprojection into prev frame (not same pixel).
// • Spatial reuse samples 8-neighborhood from PREVIOUS frame only (single-pass safe).
// • Mixture-PDF + UCW ("big-W") retained; visibility tested once using normal-offset.
// • No inner/local functions; helpers are static in class scope.
// ==============================
using System;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace ILGPU_Raytracing.Engine
{
    // --- SoA reservoir view ---
    public struct GpuReservoirSoA
    {
        public ArrayView<Float3> L;
        public ArrayView<Float3> wi;
        public ArrayView<float> pdf;
        public ArrayView<float> w;
        public ArrayView<float> wSum;
        public ArrayView<int> m;
        public ArrayView<int> lightId;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Reservoir Read(int index)
        {
            Reservoir r;
            r.L = L[index]; r.wi = wi[index]; r.pdf = pdf[index];
            r.w = w[index]; r.wSum = wSum[index]; r.m = m[index]; r.lightId = lightId[index];
            return r;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Write(int index, in Reservoir r)
        {
            L[index] = r.L; wi[index] = r.wi; pdf[index] = r.pdf;
            w[index] = r.w; wSum[index] = r.wSum; lightId[index] = r.lightId;
            m[index] = r.m; // write m last (ready flag)
        }
    }

    // --- Framebuffer ---
    public struct GpuFramebuffer
    {
        public ArrayView<int> color;
        public ArrayView<float> depth;
        public ArrayView<int> objectId;
        public ArrayView<int> cameraId;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Store(int index, Float3 rgb, float z, int obj)
        {
            color[index] = PackRGBA8(rgb);
            depth[index] = z;
            objectId[index] = obj;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int PackRGBA8(Float3 c)
        {
            int R = ToByte(c.X), G = ToByte(c.Y), B = ToByte(c.Z);
            return (255 << 24) | (R << 16) | (G << 8) | B;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToByte(float x)
        {
            float c = XMath.Min(1f, XMath.Max(0f, x));
            return (int)(255.99f * c);
        }
    }

    // --- GBuffer ---
    public struct GpuGBuffer
    {
        public ArrayView<Float3> worldPos;
        public ArrayView<Float3> normalWS;
        public ArrayView<Float3> baseColor;
        public ArrayView<int> matId;
        public ArrayView<int> objId;
        public ArrayView<int> hitMask;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StoreHit(int index, Float3 posWS, Float3 nWS, Float3 albedo, int packedMat, int oid)
        {
            hitMask[index] = 1;
            worldPos[index] = posWS;
            normalWS[index] = nWS;
            baseColor[index] = albedo;
            matId[index] = packedMat;
            objId[index] = oid;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void StoreMiss(int index, Ray primary)
        {
            hitMask[index] = 0;
            worldPos[index] = primary.origin + primary.dir * 1e6f;
            normalWS[index] = new Float3(0f, 1f, 0f);
            baseColor[index] = new Float3(0f, 0f, 0f);
            matId[index] = -1;
            objId[index] = -1;
        }
    }

    // --- Params ---
    public struct GBufferParams
    {
        public int width, height, frame;
        public Camera cam;
        public SceneDeviceViews views;
        public GpuGBuffer gb;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Ray PrimaryRay(int index)
        {
            int x = index % width, y = index / width;
            float u = ((float)x + 0.5f) / (float)XMath.Max(1, width);
            float v = ((float)y + 0.5f) / (float)XMath.Max(1, height);
            return Ray.GenerateRay(cam, u, v);
        }
    }

    public struct IntegratorParams
    {
        public int width, height, frame;
        public Camera cam;
        public Camera prevCam; // NEW: for temporal reprojection
        public SceneDeviceViews views;
        public GpuGBuffer gb;
        public GpuFramebuffer fb;
        public Float3 dirLightDir, dirLightRadiance;
        public Float3 skyTintTop, skyTintBottom;
        public int debugCamSeq;

        public GpuReservoirSoA resPrev; // READ-ONLY (prev frame, after spatial)
        public GpuReservoirSoA resCur;  // WRITE-ONLY in this pass

        public int enableTemporalReuse, enableSpatialReuse, rngLockNoise;
        public int spp;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Float3 PrimaryRayDir(int index)
        {
            int x = index % width, y = index / width;
            float u = ((float)x + 0.5f) / (float)XMath.Max(1, width);
            float v = ((float)y + 0.5f) / (float)XMath.Max(1, height);
            return Ray.GenerateRay(cam, u, v).dir;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Float3 ViewDirFromCam(Float3 posWS) => Float3.Normalize(posWS - cam.origin);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DistanceFromCamera(Float3 posWS)
        {
            Float3 d = posWS - cam.origin;
            return XMath.Sqrt(d.X * d.X + d.Y * d.Y + d.Z * d.Z);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Float3 SkyWeighted(Float3 dir)
        {
            float tbg = 0.5f * (dir.Y + 1.0f);
            return skyTintBottom * (1f - tbg) + skyTintTop * tbg;
        }
    }

    public struct Reservoir
    {
        public Float3 L, wi;
        public float pdf;     // selection pdf (mixture)
        public float w;       // score at selection
        public float wSum;    // sum of scores
        public int m;         // multiplicity (candidates seen)
        public int lightId;   // 1 = env/BRDF, 2 = directional
    }

    public static class RTRay
    {
        private const float PI = 3.14159265358979323846f;
        private const float INV_PI = 0.31830988618379067154f;
        private const float EPS_N = 0.0025f;
        private const float EPS_MIN = 1e-6f;

        public static void PrimaryVisibilityKernel(Index1D index, GBufferParams p)
        {
            long length = p.gb.worldPos.Length;
            if (index >= length) return;

            var wray = p.PrimaryRay(index);
            float t; Float3 n; Float3 albedo; int objId; int shade; float ior;
            bool hit = p.views.TraceClosest(wray, out t, out n, out albedo, out objId, out shade, out ior);

            if (!hit) { p.gb.StoreMiss(index, wray); return; }
            Float3 posWS = wray.origin + wray.dir * t;
            int packedMat = (shade & 0xFFFF) | (FloatToI16(ior) << 16);
            p.gb.StoreHit(index, posWS, n, albedo, packedMat, objId);
        }

        public static void PathTraceKernel(Index1D index, IntegratorParams k, SpecializedValue<int> MaxDepth)
        {
            if (index >= k.fb.color.Length) return;
            if (index == 0 && k.fb.cameraId.Length > 0) k.fb.cameraId[0] = k.debugCamSeq;

            Float3 Lframe = default;

            for (int s = 0; s < XMath.Max(1, k.spp); s++)
            {
                RNG rng = RNG.CreateFromIndex1D(index, k.width, k.height, k.frame, (uint)s, 0xC0FFEEu, k.rngLockNoise);

                if (k.gb.hitMask[index] == 0)
                {
                    var vdir = k.PrimaryRayDir(index);
                    Lframe += SafeColor(k.SkyWeighted(vdir));
                    continue;
                }

                Float3 pos = k.gb.worldPos[index];
                Float3 nrm = Float3.Normalize(k.gb.normalWS[index]);
                Float3 alb = k.gb.baseColor[index];
                int packedMat = k.gb.matId[index];
                int shade = packedMat & 0xFFFF;
                float ior = I16ToFloat((packedMat >> 16) & 0xFFFF);

                Float3 Li = new Float3(0f, 0f, 0f);
                Float3 throughput = new Float3(1f, 1f, 1f);
                Float3 I = k.ViewDirFromCam(pos);
                bool wroteReservoir = false;

                for (int depth = 0; depth < MaxDepth; depth++)
                {
                    if (shade == Sphere.SHADING_MIRROR)
                    {
                        Float3 dirR = Reflect(I, nrm);
                        Ray ray = MakeRayWithNormalOffset(pos, nrm, dirR, EPS_N);
                        throughput = throughput * alb;

                        if (!TraceNext(k, ray, ref pos, ref nrm, ref alb, ref shade, ref ior))
                        { Li += throughput * k.SkyWeighted(ray.dir); break; }
                        I = ray.dir; continue;
                    }

                    if (shade == Sphere.SHADING_GLASS)
                    {
                        Float3 Nuse = nrm;
                        bool outside = Float3.Dot(I, nrm) < 0f;
                        if (!outside) Nuse = Nuse * -1f;
                        float etaI = outside ? 1f : (ior > 0f ? ior : 1.5f);
                        float etaT = outside ? (ior > 0f ? ior : 1.5f) : 1f;

                        Float3 dirR = Reflect(I, Nuse);
                        Float3 dirT;
                        bool refrOk = Refract(I, Nuse, etaI, etaT, out dirT);
                        float cosI = XMath.Abs(Float3.Dot(I, Nuse));
                        float Fr = SchlickFresnel(cosI, etaI, etaT);
                        float xi = rng.NextFloat();

                        Ray ray = (!refrOk || xi < Fr)
                            ? MakeRayWithNormalOffset(pos, Nuse, dirR, EPS_N)
                            : MakeRayWithNormalOffset(pos, -Nuse, dirT, EPS_N);

                        if (refrOk && xi >= Fr)
                        {
                            Float3 transTint = (alb.X == 0f && alb.Y == 0f && alb.Z == 0f) ? new Float3(1f, 1f, 1f) : alb;
                            float etaScale = (etaI * etaI) / (etaT * etaT);
                            throughput = throughput * transTint * etaScale;
                        }

                        if (!TraceNext(k, ray, ref pos, ref nrm, ref alb, ref shade, ref ior))
                        { Li += throughput * k.SkyWeighted(ray.dir); break; }
                        I = ray.dir; continue;
                    }

                    // Non-specular: ReSTIR-DI + diffuse bounce
                    {
                        Reservoir outRes;
                        if (wroteReservoir)
                        {
                            IntegratorParams kLocal = k;
                            kLocal.enableTemporalReuse = 0;
                            kLocal.enableSpatialReuse = 0;
                            Float3 direct = ReSTIR_Direct(index, kLocal, pos, nrm, alb, ref rng, out outRes);
                            Li += throughput * direct;
                        }
                        else
                        {
                            Float3 direct = ReSTIR_Direct(index, k, pos, nrm, alb, ref rng, out outRes);
                            Li += throughput * direct;
                            if (k.resCur.L.Length > index)
                            {
                                k.resCur.Write(index, outRes);
                                wroteReservoir = true;
                            }
                        }
                    }

                    // Indirect Lambertian bounce
                    {
                        Float3 wi = SampleHemisphereCosine(nrm, ref rng);
                        Ray ray = MakeRayWithNormalOffset(pos, nrm, wi, EPS_N);
                        throughput = throughput * alb;

                        if (depth >= 3)
                        {
                            float maxC = XMath.Max(throughput.X, XMath.Max(throughput.Y, throughput.Z));
                            maxC = XMath.Clamp(maxC, 0.05f, 0.98f);
                            if (rng.NextFloat() > maxC) { throughput = new Float3(0f, 0f, 0f); break; }
                            throughput = throughput * (1.0f / maxC);
                        }

                        if (!TraceNext(k, ray, ref pos, ref nrm, ref alb, ref shade, ref ior))
                        { Li += throughput * k.SkyWeighted(ray.dir); break; }
                        I = ray.dir; continue;
                    }
                }

                Lframe += SafeColor(Li);
            }

            Float3 Lout = Lframe * (1.0f / XMath.Max(1, k.spp));
            k.fb.Store(index, Lout, k.DistanceFromCamera(k.gb.worldPos[index]), k.gb.objId[index]);
        }

        // ---- ReSTIR DI ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Reservoir NewReservoirDefault()
        {
            Reservoir r;
            r.L = default; r.wi = default; r.pdf = 0f; r.w = 0f; r.wSum = 0f; r.m = 0; r.lightId = 0;
            return r;
        }

        // camera reprojection → prev-frame pixel index (or -1)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ReprojectToPrevPixel(Float3 posWS, in IntegratorParams k)
        {
            // build vector in prev camera space
            Float3 p = posWS - k.prevCam.origin;
            float x = Float3.Dot(p, k.prevCam.right);
            float y = Float3.Dot(p, k.prevCam.up);
            float z = Float3.Dot(p, k.prevCam.forward); // forward is +Z view direction

            if (z <= 1e-4f) return -1; // behind

            float tanHalfFov = XMath.Tan(0.5f * k.prevCam.fovYRadians);
            float ndcX = x / (z * tanHalfFov * k.prevCam.aspect);
            float ndcY = y / (z * tanHalfFov);

            // to pixel
            float fx = 0.5f * (ndcX + 1f) * k.width;
            float fy = 0.5f * (ndcY + 1f) * k.height;
            int px = (int)fx;
            int py = (int)fy;
            if ((uint)px >= (uint)k.width || (uint)py >= (uint)k.height) return -1;
            return py * k.width + px;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool SpatialCompatible(in IntegratorParams k, int idxA, int idxB, Float3 nA)
        {
            int objA = k.gb.objId[idxA], objB = k.gb.objId[idxB];
            if (objA == objB) return true;
            Float3 nB = Float3.Normalize(k.gb.normalWS[idxB]);
            float ndot = Float3.Dot(nA, nB);
            if (ndot < 0.85f) return false;
            float zA = k.DistanceFromCamera(k.gb.worldPos[idxA]);
            float zB = k.DistanceFromCamera(k.gb.worldPos[idxB]);
            float rel = XMath.Abs(zA - zB) / XMath.Max(1e-3f, zA);
            return rel < 0.05f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Neighbor8(int rot, int radius, out int dx0, out int dy0, out int dx1, out int dy1,
            out int dx2, out int dy2, out int dx3, out int dy3, out int dx4, out int dy4, out int dx5, out int dy5, out int dx6, out int dy6, out int dx7, out int dy7)
        {
            int r = radius;
            int RX(int x, int y, int R) => R == 0 ? x : (R == 1 ? -y : (R == 2 ? -x : y));
            int RY(int x, int y, int R) => R == 0 ? y : (R == 1 ? x : (R == 2 ? -y : -x));
            dx0 = RX(-r, 0, rot); dy0 = RY(-r, 0, rot);
            dx1 = RX(r, 0, rot); dy1 = RY(r, 0, rot);
            dx2 = RX(0, -r, rot); dy2 = RY(0, -r, rot);
            dx3 = RX(0, r, rot); dy3 = RY(0, r, rot);
            dx4 = RX(-r, -r, rot); dy4 = RY(-r, -r, rot);
            dx5 = RX(r, -r, rot); dy5 = RY(r, -r, rot);
            dx6 = RX(-r, r, rot); dy6 = RY(-r, r, rot);
            dx7 = RX(r, r, rot); dy7 = RY(r, r, rot);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ReservoirUpdate(ref Reservoir r, Float3 wi, float pdfSel, Float3 Li, float scoreS, int multiplicity, int lightId, ref RNG rng)
        {
            float add = scoreS;
            float newSum = r.wSum + add;
            float acceptP = (newSum > 0f) ? add / newSum : 0f;
            if (rng.NextFloat() < acceptP)
            {
                r.wi = wi; r.pdf = pdfSel; r.L = Li; r.w = scoreS; r.lightId = lightId;
            }
            r.wSum = newSum;
            r.m = r.m + XMath.Max(1, multiplicity);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ImportFromPrevReservoir(
            int prevIdx, int curIdx, in IntegratorParams k,
            Float3 n, Float3 albedo, float mixLocal, float mixDelta,
            ref RNG rng, ref Reservoir r)
        {
            if (prevIdx < 0 || k.resPrev.L.Length <= prevIdx) return;
            if (!SpatialCompatible(k, curIdx, prevIdx, n)) return;

            Reservoir pr = k.resPrev.Read(prevIdx);
            if (!(pr.m > 0 && pr.w > 0f && pr.wSum > 0f)) return;

            Float3 wi = pr.wi;
            int lid = pr.lightId == 2 ? 2 : 1;
            Float3 LiImp = (lid == 2) ? k.dirLightRadiance : k.SkyWeighted(wi);

            float nl = XMath.Max(0f, Float3.Dot(n, wi));
            float pdfHere = (lid == 2)
                ? XMath.Max(EPS_MIN, mixDelta)
                : XMath.Max(EPS_MIN, CosHemispherePdf(n, wi) * mixLocal);

            Float3 f_over_p = albedo * LiImp * ((nl / pdfHere) * INV_PI);
            float sHere = Luminance(f_over_p);

            float Wsrc = pr.wSum / (XMath.Max(1, pr.m) * XMath.Max(EPS_MIN, pr.w));
            float eff = sHere * Wsrc;

            ReservoirUpdate(ref r, wi, pdfHere, LiImp, eff, 1, lid, ref rng);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 ReSTIR_Direct(
            int index, in IntegratorParams k,
            Float3 pos, Float3 n, Float3 albedo,
            ref RNG rng, out Reservoir outRes)
        {
            const int LocalCandidates = 8;
            const int DeltaCandidates = 1;
            const int TotalNew = LocalCandidates + DeltaCandidates;
            float mixLocal = (float)LocalCandidates / (float)TotalNew;
            float mixDelta = (float)DeltaCandidates / (float)TotalNew;

            Reservoir r = NewReservoirDefault();

            // (1) Local BRDF candidates (no vis in weights)
            for (int i = 0; i < LocalCandidates; i++)
            {
                Float3 wi = SampleHemisphereCosine(n, ref rng);
                float nl = XMath.Max(0f, Float3.Dot(n, wi));
                float pdfLocal = XMath.Max(EPS_MIN, CosHemispherePdf(n, wi));
                float pdfSel = XMath.Max(EPS_MIN, pdfLocal * mixLocal);
                Float3 LiLoc = k.SkyWeighted(wi);
                Float3 f_over_p = albedo * LiLoc * ((nl / pdfSel) * INV_PI);
                float s = Luminance(f_over_p);
                ReservoirUpdate(ref r, wi, pdfSel, LiLoc, s, 1, 1, ref rng);
            }

            // (2) Directional delta candidate
            {
                Float3 wi = Float3.Normalize(k.dirLightDir);
                float nl = XMath.Max(0f, Float3.Dot(n, wi));
                float pdfSel = XMath.Max(EPS_MIN, mixDelta);
                Float3 LiDir = k.dirLightRadiance;
                Float3 f_over_p = albedo * LiDir * ((nl / pdfSel) * INV_PI);
                float s = Luminance(f_over_p);
                ReservoirUpdate(ref r, wi, pdfSel, LiDir, s, 1, 2, ref rng);
            }

            // (3) Temporal reuse with reprojection into prev frame
            if (k.enableTemporalReuse != 0)
            {
                int prevIdx = ReprojectToPrevPixel(pos, k);
                if (prevIdx >= 0) // valid reprojection
                {
                    ImportFromPrevReservoir(prevIdx, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                }
            }

            // (4) Spatial reuse from PREVIOUS frame only (8 neighbors)
            if (k.enableSpatialReuse != 0)
            {
                uint h = Hash3((uint)index, (uint)k.frame, 0xB31F5AB1u);
                int rot = (int)(h & 3u);
                int radius = 1 + (int)((h >> 2) & 1u);
                int x0 = index % k.width, y0 = index / k.width;

                Neighbor8(rot, radius,
                    out int dx0, out int dy0, out int dx1, out int dy1,
                    out int dx2, out int dy2, out int dx3, out int dy3,
                    out int dx4, out int dy4, out int dx5, out int dy5,
                    out int dx6, out int dy6, out int dx7, out int dy7);

                int n0 = (uint)(x0 + dx0) < (uint)k.width && (uint)(y0 + dy0) < (uint)k.height ? (y0 + dy0) * k.width + (x0 + dx0) : -1;
                int n1 = (uint)(x0 + dx1) < (uint)k.width && (uint)(y0 + dy1) < (uint)k.height ? (y0 + dy1) * k.width + (x0 + dx1) : -1;
                int n2 = (uint)(x0 + dx2) < (uint)k.width && (uint)(y0 + dy2) < (uint)k.height ? (y0 + dy2) * k.width + (x0 + dx2) : -1;
                int n3 = (uint)(x0 + dx3) < (uint)k.width && (uint)(y0 + dy3) < (uint)k.height ? (y0 + dy3) * k.width + (x0 + dx3) : -1;
                int n4 = (uint)(x0 + dx4) < (uint)k.width && (uint)(y0 + dy4) < (uint)k.height ? (y0 + dy4) * k.width + (x0 + dx4) : -1;
                int n5 = (uint)(x0 + dx5) < (uint)k.width && (uint)(y0 + dy5) < (uint)k.height ? (y0 + dy5) * k.width + (x0 + dx5) : -1;
                int n6 = (uint)(x0 + dx6) < (uint)k.width && (uint)(y0 + dy6) < (uint)k.height ? (y0 + dy6) * k.width + (x0 + dx6) : -1;
                int n7 = (uint)(x0 + dx7) < (uint)k.width && (uint)(y0 + dy7) < (uint)k.height ? (y0 + dy7) * k.width + (x0 + dx7) : -1;

                ImportFromPrevReservoir(n0, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n1, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n2, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n3, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n4, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n5, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n6, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
                ImportFromPrevReservoir(n7, index, k, n, albedo, mixLocal, mixDelta, ref rng, ref r);
            }

            // (5) Final shading of selected sample with single visibility test
            Float3 contrib = new Float3(0f, 0f, 0f);
            if (r.m > 0 && r.wSum > 0f && r.w > 0f)
            {
                Float3 wiSel = r.wi;
                int lidSel = r.lightId == 2 ? 2 : 1;

                float nlSel = XMath.Max(0f, Float3.Dot(n, wiSel));
                if (nlSel > 0f && Visible(k, pos, n, wiSel))
                {
                    float mixLocal2 = (float)LocalCandidates / (float)(LocalCandidates + DeltaCandidates);
                    float mixDelta2 = (float)DeltaCandidates / (float)(LocalCandidates + DeltaCandidates);
                    float pdfSel = (lidSel == 2)
                        ? XMath.Max(EPS_MIN, mixDelta2)
                        : XMath.Max(EPS_MIN, CosHemispherePdf(n, wiSel) * mixLocal2);

                    Float3 LiSel = (lidSel == 2) ? k.dirLightRadiance : k.SkyWeighted(wiSel);
                    Float3 f_over_p = albedo * LiSel * ((nlSel / pdfSel) * INV_PI);
                    float W = r.wSum / (float)XMath.Max(1, r.m) / XMath.Max(EPS_MIN, r.w);
                    contrib = f_over_p * W;
                }
            }

            outRes = r;
            return contrib;
        }

        // ---- Shared math ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 InvDir(Float3 d) =>
            new Float3(1f / (d.X != 0f ? d.X : 1e-8f), 1f / (d.Y != 0f ? d.Y : 1e-8f), 1f / (d.Z != 0f ? d.Z : 1e-8f));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray MakeRayWithNormalOffset(Float3 origin, Float3 n, Float3 dir, float epsN)
        {
            Float3 d = Float3.Normalize(dir);
            float s = Float3.Dot(n, d) >= 0f ? 1f : -1f;
            Float3 o = origin + n * (epsN * s);
            return new Ray { origin = o, dir = d, invDir = InvDir(d) };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Reflect(Float3 I, Float3 N) => I - N * (2f * Float3.Dot(I, N));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Refract(Float3 I, Float3 N, float etaI, float etaT, out Float3 T)
        {
            float eta = etaI / etaT;
            float cosI = -Float3.Dot(I, N);
            float k = 1f - eta * eta * (1f - cosI * cosI);
            if (k < 0f) { T = default; return false; }
            T = Float3.Normalize(I * eta + N * (eta * cosI - XMath.Sqrt(k)));
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SchlickFresnel(float cos, float etaI, float etaT)
        {
            float r0 = (etaI - etaT) / (etaI + etaT);
            r0 = r0 * r0;
            float oneMinusCos = 1f - cos;
            float oneMinusCos2 = oneMinusCos * oneMinusCos;
            float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;
            return r0 + (1f - r0) * oneMinusCos5;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SampleHemisphereCosine(Float3 n, ref RNG rng)
        {
            float r1 = rng.NextFloat(), r2 = rng.NextFloat();
            float phi = 2f * PI * r1;
            float cosTheta = XMath.Sqrt(1f - r2);
            float sinTheta = XMath.Sqrt(r2);
            float x = XMath.Cos(phi) * sinTheta;
            float y = XMath.Sin(phi) * sinTheta;
            float z = cosTheta;
            OrthonormalBasis(n, out var t, out var b);
            Float3 v = t * x + b * y + n * z;
            return Float3.Normalize(v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void OrthonormalBasis(Float3 n, out Float3 t, out Float3 b)
        {
            Float3 up = XMath.Abs(n.Y) < 0.999f ? new Float3(0f, 1f, 0f) : new Float3(1f, 0f, 0f);
            t = Float3.Normalize(Float3.Cross(up, n));
            b = Float3.Cross(n, t);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int FloatToI16(float x)
        {
            float cl = XMath.Max(0f, XMath.Min(65535f, x * 1000f));
            return (int)cl & 0xFFFF;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float I16ToFloat(int v) => (float)v / 1000f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Visible(in IntegratorParams k, Float3 origin, Float3 n, Float3 wi)
        {
            float nl = Float3.Dot(n, wi);
            if (nl <= 0f) return false;
            Ray s = MakeRayWithNormalOffset(origin, n, wi, EPS_N);
            return !k.views.ShadowOcclusion(s, 1e29f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Luminance(Float3 c) => 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CosHemispherePdf(Float3 n, Float3 wi)
        {
            float nl = XMath.Max(0f, Float3.Dot(n, wi));
            return nl * INV_PI;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Hash(uint x)
        {
            x ^= x >> 17; x *= 0xed5ad4bbu; x ^= x >> 11; x *= 0xac4c1b51u; x ^= x >> 15; x *= 0x31848babu; x ^= x >> 14;
            return x;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Hash3(uint a, uint b, uint c) => Hash(a ^ Hash(b ^ Hash(c)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SafeColor(Float3 c)
        {
            float x = (float)(double.IsFinite(c.X) ? c.X : 0f);
            float y = (float)(double.IsFinite(c.Y) ? c.Y : 0f);
            float z = (float)(double.IsFinite(c.Z) ? c.Z : 0f);
            x = XMath.Min(1e6f, XMath.Max(-1e6f, x));
            y = XMath.Min(1e6f, XMath.Max(-1e6f, y));
            z = XMath.Min(1e6f, XMath.Max(-1e6f, z));
            return new Float3(x, y, z);
        }

        // Trace helper
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraceNext(in IntegratorParams k, in Ray ray,
                                      ref Float3 pos, ref Float3 nrm, ref Float3 alb, ref int shade, ref float ior)
        {
            float t; Float3 n2; Float3 alb2; int obj2; int shade2; float ior2;
            bool hit2 = k.views.TraceClosest(ray, out t, out n2, out alb2, out obj2, out shade2, out ior2);
            if (!hit2) return false;
            pos = ray.origin + ray.dir * t;
            nrm = Float3.Normalize(n2);
            alb = alb2;
            shade = shade2;
            ior = ior2;
            return true;
        }
    }
}
