// ==============================
// File: Engine/RTRay.cs
// Kernel + RT helpers in RTRay; alpha cutout is driven ONLY by map_d (alpha mask)
// - Diffuse A channel is ignored for cutouts
// - No "cheats" or fallbacks
// - Perf: SpecializedValue<> budgets to enable DCE; two-tier alpha in shadow any-hit
// ==============================

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace ILGPU_Raytracing.Engine
{
    public struct SceneDeviceViews
    {
        public ArrayView<TLASNode> tlasNodes;
        public ArrayView<int> tlasInstanceIndices;
        public ArrayView<InstanceRecord> instances;
        public ArrayView<BLASNode> blasNodes;
        public ArrayView<int> spherePrimIdx;
        public ArrayView<Sphere> spheres;
        public ArrayView<int> triPrimIdx;
        public ArrayView<Float3> meshPositions;
        public ArrayView<MeshTri> meshTris;
        public ArrayView<Float2> meshTexcoords;
        public ArrayView<MeshTriUV> meshTriUVs;
        public ArrayView<int> triMatIndex;
        public ArrayView<MaterialRecord> materials;
        public ArrayView<RGBA32> texels;
        public ArrayView<TexInfo> texInfos;
    }

    public struct KernelParams
    {
        public int width;
        public int height;
        public int frame;

        // multi-bounce budgets (non-stochastic, deterministic chaining)
        public int diffuseRays;     // safety cap
        public int reflectionRays;  // perfect reflection bounces
        public int refractionRays;  // refraction bounces
        public int shadowRays;      // visibility rays per light direction (0 disables hard shadows)

        public Camera cam;
        public SceneDeviceViews views;
        public GpuFramebuffer fb;
    }

    public static class RTRay
    {
        public static void RayTraceKernel(Index1D index, KernelParams k, SpecializedValue<int> DiffuseBudget, SpecializedValue<int> ReflBudget, SpecializedValue<int> RefrBudget, SpecializedValue<int> ShadowOn)
        {
            if (index >= k.fb.color.Length) return;

            int x = index % k.width;
            int y = index / k.width;

            float u = ((float)x + 0.5f) / (float)XMath.Max(1, k.width);
            float v = ((float)y + 0.5f) / (float)XMath.Max(1, k.height);

            Ray wray = Ray.GenerateRay(k.cam, u, v);

            Float3 color;
            float depth;
            int objectId;

            // Fast path: if all budgets are zero, do a single closest-hit + direct lighting (+ optional hard shadow).
            if (DiffuseBudget.Value == 0 && ReflBudget.Value == 0 && RefrBudget.Value == 0)
            {
                ShadePath_Simple(k.views, wray, ShadowOn.Value != 0 ? 1 : 0, out color, out depth, out objectId);
            }
            else
            {
                ShadePath(k.views, wray, DiffuseBudget.Value, ReflBudget.Value, RefrBudget.Value, ShadowOn.Value != 0 ? 1 : 0, out color, out depth, out objectId);
            }

            int R = ToByte(color.X);
            int G = ToByte(color.Y);
            int B = ToByte(color.Z);
            k.fb.color[index] = (255 << 24) | (R << 16) | (G << 8) | B;
            k.fb.depth[index] = depth;
            k.fb.objectId[index] = objectId;
        }

        // Deterministic multi-bounce integrator with hard-shadow next-event estimation toward a single directional light.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ShadePath(SceneDeviceViews views, Ray wray, int diffuseBudget, int reflectionBudget, int refractionBudget, int shadowEnabled, out Float3 outColor, out float outDepth, out int outObjId)
        {
            Float3 accum = new Float3(0f, 0f, 0f);
            Float3 throughput = new Float3(1f, 1f, 1f);

            Ray ray = wray;

            bool firstHitSet = false;
            float firstDepth = float.PositiveInfinity;
            int firstObjId = -1;

            int maxSteps = 2 + XMath.Max(0, diffuseBudget) + XMath.Max(0, reflectionBudget) + XMath.Max(0, refractionBudget);
            if (maxSteps < 2) maxSteps = 2;
            if (maxSteps > 64) maxSteps = 64;

            Float3 lightDir = Float3.Normalize(new Float3(-0.4f, 1f, -0.2f));

            for (int step = 0; step < maxSteps; step++)
            {
                float t; Float3 n; Float3 albedo; int objId; int shade; float ior;
                bool hit = TraceClosest(views, ray, out t, out n, out albedo, out objId, out shade, out ior);

                if (!hit)
                {
                    Float3 sky = SkyColor(ray.dir);
                    accum = accum + throughput * sky;
                    break;
                }

                if (!firstHitSet)
                {
                    firstHitSet = true;
                    firstDepth = t;
                    firstObjId = objId;
                }

                if (shade == Sphere.SHADING_MIRROR && reflectionBudget > 0)
                {
                    Float3 rdir = Reflect(ray.dir, n);
                    ray = MakeRayOffset(ray, t, n, rdir, 0.001f);
                    throughput = throughput * albedo;
                    reflectionBudget--;
                    continue;
                }
                else if (shade == Sphere.SHADING_GLASS && (refractionBudget > 0 || reflectionBudget > 0))
                {
                    Float3 nn = n;
                    float n1 = 1f;
                    float n2 = ior > 0f ? ior : 1.5f;
                    float cosI = Float3.Dot(ray.dir, nn);
                    bool outside = cosI < 0f;
                    if (!outside) nn = nn * -1f;
                    float etaI = outside ? n1 : n2;
                    float etaT = outside ? n2 : n1;
                    Float3 refrDir;
                    bool refrOk = Refract(ray.dir, nn, etaI, etaT, out refrDir);
                    float cos = XMath.Abs(Float3.Dot(ray.dir, nn));
                    float Fr = SchlickFresnel(cos, etaI, etaT);

                    if (reflectionBudget > 0)
                    {
                        Float3 rdir = Reflect(ray.dir, nn);
                        Ray rr = MakeRayOffset(ray, t, nn, rdir, 0.001f);
                        Float3 rc; float rt; int ro;
                        ShadeOneBounceDiffuse(views, rr, lightDir, shadowEnabled, out rc, out rt, out ro);
                        accum = accum + throughput * albedo * rc * Fr;
                        reflectionBudget--;
                    }

                    if (refrOk && refractionBudget > 0)
                    {
                        ray = MakeRayOffsetAlongDir(ray, t, refrDir, 0.002f);
                        throughput = throughput * albedo * (1f - Fr);
                        refractionBudget--;
                        continue;
                    }
                    else
                    {
                        Float3 rdir = Reflect(ray.dir, nn);
                        ray = MakeRayOffset(ray, t, nn, rdir, 0.001f);
                        throughput = throughput * albedo;
                        if (reflectionBudget > 0) { reflectionBudget--; continue; }

                        Float3 rc; float rt; int ro;
                        ShadeOneBounceDiffuse(views, ray, lightDir, shadowEnabled, out rc, out rt, out ro);
                        accum = accum + throughput * rc;
                        break;
                    }
                }
                else
                {
                    float ndotl = XMath.Max(0f, Float3.Dot(n, lightDir));
                    float vis = 1f;
                    if (shadowEnabled != 0 && ndotl > 0f)
                    {
                        // single hard shadow sample toward the directional light
                        Ray sray = MakeRayOffsetAlongDir(ray, t, lightDir, 0.0015f);
                        vis = ShadowOcclusion(views, sray, 1e29f) ? 0f : 1f;
                    }
                    Float3 local = albedo * (0.15f + 0.85f * ndotl * vis);
                    accum = accum + throughput * local;
                    break;
                }
            }

            if (!firstHitSet)
            {
                Float3 sky = SkyColor(wray.dir);
                outColor = sky;
                outDepth = float.PositiveInfinity;
                outObjId = -1;
            }
            else
            {
                outColor = accum;
                outDepth = firstDepth;
                outObjId = firstObjId;
            }
        }

        // Single-bounce simplified shading; adds optional hard shadow toward directional light.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ShadePath_Simple(SceneDeviceViews views, Ray wray, int shadowEnabled, out Float3 outColor, out float outDepth, out int outObjId)
        {
            Float3 lightDir = Float3.Normalize(new Float3(-0.4f, 1f, -0.2f));
            float t; Float3 n; Float3 albedo; int objId; int shade; float ior;
            bool hit = TraceClosest(views, wray, out t, out n, out albedo, out objId, out shade, out ior);
            if (!hit)
            {
                outColor = SkyColor(wray.dir);
                outDepth = float.PositiveInfinity;
                outObjId = -1;
                return;
            }
            float ndotl = XMath.Max(0f, Float3.Dot(n, lightDir));
            float vis = 1f;
            if (shadowEnabled != 0 && ndotl > 0f)
            {
                Ray sray = MakeRayOffsetAlongDir(wray, t, lightDir, 0.0015f);
                vis = ShadowOcclusion(views, sray, 1e29f) ? 0f : 1f;
            }
            outColor = albedo * (0.15f + 0.85f * ndotl * vis);
            outDepth = t;
            outObjId = objId;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ShadeOneBounceDiffuse(SceneDeviceViews views, Ray wray, Float3 lightDir, int shadowEnabled, out Float3 outColor, out float outDepth, out int outObjId)
        {
            float closestT; Float3 bestNormal; Float3 bestAlbedo; int bestObjId; int bestShade; float bestIor;
            bool hit = TraceClosest(views, wray, out closestT, out bestNormal, out bestAlbedo, out bestObjId, out bestShade, out bestIor);
            if (hit)
            {
                float ndotl = XMath.Max(0f, Float3.Dot(bestNormal, lightDir));
                float vis = 1f;
                if (shadowEnabled != 0 && ndotl > 0f)
                {
                    Ray sray = MakeRayOffsetAlongDir(wray, closestT, lightDir, 0.0015f);
                    vis = ShadowOcclusion(views, sray, 1e29f) ? 0f : 1f;
                }
                outColor = bestAlbedo * (0.15f + 0.85f * ndotl * vis);
                outDepth = closestT;
                outObjId = bestObjId;
            }
            else
            {
                outColor = SkyColor(wray.dir);
                outDepth = float.PositiveInfinity;
                outObjId = -1;
            }
        }

        // -------- Any-hit BVH traversal for hard shadowing (respects alpha cutouts) --------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShadowOcclusion(SceneDeviceViews views, Ray srayWorld, float tMaxWorld)
        {
            int cur = 0;
            while (cur != -1)
            {
                TLASNode n = views.tlasNodes[cur];
                if (IntersectAABB(srayWorld, n.boundsMin, n.boundsMax, 0.001f, tMaxWorld))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int instIndex = views.tlasInstanceIndices[i];
                            InstanceRecord inst = views.instances[instIndex];
                            Ray srayObj = TransformRay(inst.worldToObject, srayWorld);
                            float scale = inst.uniformScale > 0f ? inst.uniformScale : 1f;
                            float tMaxObj = tMaxWorld * scale;

                            bool blocked = (inst.type == BlasType.SphereSet)
                                ? AnyHit_Sphere(views, srayObj, inst.blasRoot, inst.blasRoot + inst.blasNodeCount, views.spherePrimIdx, views.spheres, tMaxObj)
                                : AnyHit_Tri_Textured(views, srayObj, inst, inst.blasRoot, inst.blasRoot + inst.blasNodeCount, views.triPrimIdx, views.meshPositions, views.meshTris, views.meshTexcoords, views.meshTriUVs, views.triMatIndex, views.materials, views.texels, views.texInfos, tMaxObj);

                            if (blocked) return true;
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool AnyHit_Sphere(SceneDeviceViews views, Ray rayObj, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Sphere> spheres, float tMaxObj)
        {
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = views.blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tMaxObj))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int prim = primIndices[i];
                            float t; Float3 _n;
                            if (IntersectSphere(rayObj, spheres[prim], out t, out _n))
                            {
                                if (t > 0.001f && t < tMaxObj) return true;
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool AnyHit_Tri_Textured(SceneDeviceViews views, Ray rayObj, InstanceRecord inst, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, ArrayView<Float2> texcoords, ArrayView<MeshTriUV> triUVs, ArrayView<int> triMatIndex, ArrayView<MaterialRecord> materials, ArrayView<RGBA32> texels, ArrayView<TexInfo> texInfos, float tMaxObj)
        {
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = views.blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tMaxObj))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int triIndex = primIndices[i];
                            MeshTri tri = tris[triIndex];
                            Float3 v0 = positions[tri.i0];
                            Float3 v1 = positions[tri.i1];
                            Float3 v2 = positions[tri.i2];

                            float t; Float3 nn; float bu; float bv;
                            if (IntersectTriangleMT_Bary(rayObj, v0, v1, v2, out t, out nn, out bu, out bv))
                            {
                                if (t <= 0.001f || t >= tMaxObj) continue;

                                int midx = triMatIndex[triIndex];
                                MaterialRecord mat = materials[midx];

                                // Alpha cutout check for shadow rays (ONLY map_d) with two-tier sampling
                                if (mat.HasAlphaMap != 0 && mat.AlphaTexIndex >= 0 && mat.AlphaTexIndex < texInfos.Length)
                                {
                                    var tuv = triUVs[triIndex];
                                    Float2 t0 = texcoords[tuv.t0];
                                    Float2 t1 = texcoords[tuv.t1];
                                    Float2 t2 = texcoords[tuv.t2];
                                    float w = 1f - bu - bv;
                                    float uu = t0.X * w + t1.X * bu + t2.X * bv;
                                    float vv = t0.Y * w + t1.Y * bu + t2.Y * bv;

                                    // Fast path: single nearest sample
                                    float aPoint = SampleMaskPoint(texels, texInfos[mat.AlphaTexIndex], uu, vv);
                                    float cutoff = mat.AlphaCutoff;
                                    const float Band = 0.10f; // hysteresis band to avoid falling into bilinear often

                                    if (aPoint < cutoff - Band) { continue; }      // definitely transparent
                                    if (aPoint >= cutoff + Band) { return true; }  // definitely opaque

                                    // Slow path: bilinear for fence cases
                                    float aLin = SampleMaskLinear(texels, texInfos[mat.AlphaTexIndex], uu, vv);
                                    if (aLin < cutoff) { continue; }
                                }

                                return true; // any opaque hit blocks the light
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        // -------- Closest-hit path (unchanged except where noted) --------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraceClosest(SceneDeviceViews views, Ray wray, out float closestT, out Float3 bestNormal, out Float3 bestAlbedo, out int bestObjId, out int bestShade, out float bestIor)
        {
            closestT = 1e30f; bestNormal = default; bestAlbedo = new Float3(1f, 1f, 1f); bestObjId = -1; bestShade = 0; bestIor = 1f;
            int cur = 0;
            while (cur != -1)
            {
                TLASNode n = views.tlasNodes[cur];
                if (IntersectAABB(wray, n.boundsMin, n.boundsMax, 0.001f, closestT))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int instIndex = views.tlasInstanceIndices[i];
                            InstanceRecord inst = views.instances[instIndex];
                            Ray iray = TransformRay(inst.worldToObject, wray);
                            float scale = inst.uniformScale > 0f ? inst.uniformScale : 1f;
                            bool hit;
                            float tObjClosest; Float3 normalObj; Float3 albedo; int triLocal; float bu; float bv; int shade; float ior;
                            int blasStart = inst.blasRoot;
                            int blasEnd = blasStart + inst.blasNodeCount;
                            if (inst.type == BlasType.SphereSet)
                            {
                                triLocal = -1; bu = 0f; bv = 0f; shade = 0; ior = 1f;
                                hit = TraverseBLAS_Sphere(iray, views.blasNodes, blasStart, blasEnd, views.spherePrimIdx, views.spheres, views.texels, views.texInfos, out tObjClosest, out normalObj, out albedo, out shade, out ior);
                            }
                            else
                            {
                                shade = 0; ior = 1f;
                                hit = TraverseBLAS_Tri_Textured(iray, views.blasNodes, blasStart, blasEnd, views.triPrimIdx, views.meshPositions, views.meshTris, views.meshTexcoords, views.meshTriUVs, views.triMatIndex, views.materials, views.texels, views.texInfos, out tObjClosest, out normalObj, out albedo, out triLocal, out bu, out bv);
                            }
                            if (hit)
                            {
                                float tWorld = tObjClosest / scale;
                                if (tWorld < closestT)
                                {
                                    closestT = tWorld;
                                    bestNormal = Float3.Normalize(TransformVector(inst.objectToWorld, normalObj));
                                    bestAlbedo = albedo;
                                    bestObjId = triLocal;
                                    bestShade = shade;
                                    bestIor = ior;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else
                    {
                        cur = n.left;
                    }
                }
                else
                {
                    cur = n.skipIndex;
                }
            }
            return closestT < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraverseBLAS_Sphere(Ray rayObj, ArrayView<BLASNode> blasNodes, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Sphere> spheres, ArrayView<RGBA32> texels, ArrayView<TexInfo> texInfos, out float tClosest, out Float3 nObj, out Float3 albedo, out int shading, out float ior)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(1f, 1f, 1f); shading = 0; ior = 1f;
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tClosest))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int prim = primIndices[i];
                            float t; Float3 nn;
                            if (IntersectSphere(rayObj, spheres[prim], out t, out nn))
                            {
                                if (t > 0.001f && t < tClosest)
                                {
                                    tClosest = t;
                                    nObj = nn;
                                    var s = spheres[prim];
                                    Float3 kd = s.material.Kd;
                                    Float3 col = (kd.X == 0f && kd.Y == 0f && kd.Z == 0f) ? s.albedo : kd;
                                    if (s.material.HasDiffuseMap != 0 && s.material.DiffuseTexIndex >= 0 && s.material.DiffuseTexIndex < texInfos.Length)
                                    {
                                        float PI = 3.14159265358979323846f;
                                        float u = 0.5f + XMath.Atan2(nn.Z, nn.X) / (2f * PI);
                                        float v = XMath.Acos(XMath.Min(1f, XMath.Max(-1f, nn.Y))) / PI;
                                        float aTmp;
                                        col = SampleTextureLinearRGB_A(texels, texInfos[s.material.DiffuseTexIndex], u, v, out aTmp);
                                    }
                                    albedo = col;
                                    shading = s.shading;
                                    ior = s.ior > 0f ? s.ior : 1f;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return tClosest < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraverseBLAS_Tri_Textured(Ray rayObj, ArrayView<BLASNode> blasNodes, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, ArrayView<Float2> texcoords, ArrayView<MeshTriUV> triUVs, ArrayView<int> triMatIndex, ArrayView<MaterialRecord> materials, ArrayView<RGBA32> texels, ArrayView<TexInfo> texInfos, out float tClosest, out Float3 nObj, out Float3 albedo, out int triOut, out float buOut, out float bvOut)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(0.85f, 0.85f, 0.85f); triOut = -1; buOut = 0f; bvOut = 0f;
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tClosest))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int triIndex = primIndices[i];
                            MeshTri tri = tris[triIndex];
                            Float3 v0 = positions[tri.i0];
                            Float3 v1 = positions[tri.i1];
                            Float3 v2 = positions[tri.i2];

                            float t; Float3 nn; float bu; float bv;
                            if (IntersectTriangleMT_Bary(rayObj, v0, v1, v2, out t, out nn, out bu, out bv))
                            {
                                int midx = triMatIndex[triIndex];
                                MaterialRecord mat = materials[midx];

                                if (t > 0.001f && t < tClosest)
                                {
                                    var tuv = triUVs[triIndex];
                                    Float2 t0 = texcoords[tuv.t0];
                                    Float2 t1 = texcoords[tuv.t1];
                                    Float2 t2 = texcoords[tuv.t2];
                                    float w = 1f - bu - bv;
                                    float uu = t0.X * w + t1.X * bu + t2.X * bv;
                                    float vv = t0.Y * w + t1.Y * bu + t2.Y * bv;

                                    float alpha = 1f;
                                    Float3 kdCol = mat.Kd;

                                    if (mat.HasDiffuseMap != 0 && mat.DiffuseTexIndex >= 0 && mat.DiffuseTexIndex < texInfos.Length)
                                    {
                                        kdCol = SampleTextureLinear(texels, texInfos[mat.DiffuseTexIndex], uu, vv);
                                    }

                                    if (mat.HasAlphaMap != 0 && mat.AlphaTexIndex >= 0 && mat.AlphaTexIndex < texInfos.Length)
                                    {
                                        alpha = SampleMaskLinear(texels, texInfos[mat.AlphaTexIndex], uu, vv);
                                    }

                                    if (alpha < mat.AlphaCutoff)
                                    {
                                        continue;
                                    }

                                    tClosest = t;
                                    nObj = nn;
                                    if (mat.TwoSided != 0 && Float3.Dot(nObj, rayObj.dir) > 0f) nObj = nObj * -1f;
                                    albedo = kdCol;
                                    triOut = triIndex;
                                    buOut = bu;
                                    bvOut = bv;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return tClosest < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectTriangleMT_Bary(Ray ray, Float3 v0, Float3 v1, Float3 v2, out float t, out Float3 n, out float bu, out float bv)
        {
            Float3 e1 = v1 - v0;
            Float3 e2 = v2 - v0;
            Float3 p = Float3.Cross(ray.dir, e2);
            float det = Float3.Dot(e1, p);
            if (XMath.Abs(det) < 1e-8f) { t = 0f; n = default; bu = 0f; bv = 0f; return false; }
            float invDet = 1f / det;
            Float3 tv = ray.origin - v0;
            bu = Float3.Dot(tv, p) * invDet;
            if (bu < 0f || bu > 1f) { t = 0f; n = default; bv = 0f; return false; }
            Float3 q = Float3.Cross(tv, e1);
            bv = Float3.Dot(ray.dir, q) * invDet;
            if (bv < 0f || bu + bv > 1f) { t = 0f; n = default; return false; }
            t = Float3.Dot(e2, q) * invDet;
            if (t <= 0f) { n = default; return false; }
            n = Float3.Normalize(Float3.Cross(e1, e2));
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SampleTextureLinear(ArrayView<RGBA32> texels, TexInfo info, float u, float v)
        {
            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            float x = fu * (float)(info.Width - 1);
            float y = fv * (float)(info.Height - 1);
            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(info.Width - 1, x0 + 1);
            int y1 = XMath.Min(info.Height - 1, y0 + 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;

            Float3 c00 = TexelRGB(texels, info, x0, y0);
            Float3 c10 = TexelRGB(texels, info, x1, y0);
            Float3 c01 = TexelRGB(texels, info, x0, y1);
            Float3 c11 = TexelRGB(texels, info, x1, y1);
            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            return cx0 * (1f - ty) + cx1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SampleTextureLinearRGB_A(ArrayView<RGBA32> texels, TexInfo info, float u, float v, out float a)
        {
            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            float x = fu * (float)(info.Width - 1);
            float y = fv * (float)(info.Height - 1);
            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(info.Width - 1, x0 + 1);
            int y1 = XMath.Min(info.Height - 1, y0 + 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;

            RGBA32 p00 = TexelRaw(texels, info, x0, y0);
            RGBA32 p10 = TexelRaw(texels, info, x1, y0);
            RGBA32 p01 = TexelRaw(texels, info, x0, y1);
            RGBA32 p11 = TexelRaw(texels, info, x1, y1);

            Float3 c00 = new Float3(p00.R / 255f, p00.G / 255f, p00.B / 255f);
            Float3 c10 = new Float3(p10.R / 255f, p10.G / 255f, p10.B / 255f);
            Float3 c01 = new Float3(p01.R / 255f, p01.G / 255f, p01.B / 255f);
            Float3 c11 = new Float3(p11.R / 255f, p11.G / 255f, p11.B / 255f);

            float a00 = p00.A / 255f;
            float a10 = p10.A / 255f;
            float a01 = p01.A / 255f;
            float a11 = p11.A / 255f;

            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            float ax0 = a00 * (1f - tx) + a10 * tx;
            float ax1 = a01 * (1f - tx) + a11 * tx;

            a = ax0 * (1f - ty) + ax1 * ty;
            return cx0 * (1f - ty) + cx1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SampleMaskLinear(ArrayView<RGBA32> texels, TexInfo info, float u, float v)
        {
            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            float x = fu * (float)(info.Width - 1);
            float y = fv * (float)(info.Height - 1);
            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(info.Width - 1, x0 + 1);
            int y1 = XMath.Min(info.Height - 1, y0 + 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;

            float a00 = Luma01(TexelRaw(texels, info, x0, y0));
            float a10 = Luma01(TexelRaw(texels, info, x1, y0));
            float a01 = Luma01(TexelRaw(texels, info, x0, y1));
            float a11 = Luma01(TexelRaw(texels, info, x1, y1));

            float ax0 = a00 * (1f - tx) + a10 * tx;
            float ax1 = a01 * (1f - tx) + a11 * tx;
            return ax0 * (1f - ty) + ax1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SampleMaskPoint(ArrayView<RGBA32> texels, TexInfo info, float u, float v)
        {
            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            int x = (int)XMath.Round(fu * (float)(info.Width - 1));
            int y = (int)XMath.Round(fv * (float)(info.Height - 1));
            return Luma01(TexelRaw(texels, info, x, y));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Luma01(RGBA32 p)
        {
            float r = p.R * (1f / 255f);
            float g = p.G * (1f / 255f);
            float b = p.B * (1f / 255f);
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TexelRGB(ArrayView<RGBA32> texels, TexInfo info, int x, int y)
        {
            int idx = info.Offset + y * info.Width + x;
            RGBA32 p = texels[idx];
            return new Float3(p.R / 255f, p.G / 255f, p.B / 255f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static RGBA32 TexelRaw(ArrayView<RGBA32> texels, TexInfo info, int x, int y)
        {
            return texels[info.Offset + y * info.Width + x];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray TransformRay(Affine3x4 m, Ray w)
        {
            Float3 o = TransformPoint(m, w.origin);
            Float3 d = TransformVector(m, w.dir);
            Float3 inv = new Float3(1f / (d.X != 0f ? d.X : 1e-8f), 1f / (d.Y != 0f ? d.Y : 1e-8f), 1f / (d.Z != 0f ? d.Z : 1e-8f));
            return new Ray { origin = o, dir = d, invDir = inv };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformPoint(Affine3x4 m, Float3 p)
        {
            return new Float3(m.m00 * p.X + m.m01 * p.Y + m.m02 * p.Z + m.m03, m.m10 * p.X + m.m11 * p.Y + m.m12 * p.Z + m.m13, m.m20 * p.X + m.m21 * p.Y + m.m22 * p.Z + m.m23);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformVector(Affine3x4 m, Float3 v)
        {
            return new Float3(m.m00 * v.X + m.m01 * v.Y + m.m02 * v.Z, m.m10 * v.X + m.m11 * v.Y + m.m12 * v.Z, m.m20 * v.X + m.m21 * v.Y + m.m22 * v.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectSphere(Ray ray, Sphere s, out float t, out Float3 n)
        {
            Float3 oc = ray.origin - s.center;
            float a = Float3.Dot(ray.dir, ray.dir);
            float b = 2f * Float3.Dot(oc, ray.dir);
            float c = Float3.Dot(oc, oc) - s.radius * s.radius;
            float disc = b * b - 4f * a * c;
            if (disc < 0f) { t = 0f; n = default; return false; }
            float sqrtD = XMath.Sqrt(disc);
            float t0 = (-b - sqrtD) / (2f * a);
            float t1 = (-b + sqrtD) / (2f * a);
            t = t0;
            if (t < 0.001f)
            {
                t = t1;
                if (t < 0.001f) { n = default; return false; }
            }
            Float3 p = ray.origin + ray.dir * t;
            n = Float3.Normalize(p - s.center);
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectAABB(Ray ray, Float3 bmin, Float3 bmax, float tMin, float tMax)
        {
            float t1 = (bmin.X - ray.origin.X) * ray.invDir.X;
            float t2 = (bmax.X - ray.origin.X) * ray.invDir.X;
            float tmin = XMath.Min(t1, t2);
            float tmax = XMath.Max(t1, t2);

            t1 = (bmin.Y - ray.origin.Y) * ray.invDir.Y;
            t2 = (bmax.Y - ray.origin.Y) * ray.invDir.Y;
            tmin = XMath.Max(tmin, XMath.Min(t1, t2));
            tmax = XMath.Min(tmax, XMath.Max(t1, t2));

            t1 = (bmin.Z - ray.origin.Z) * ray.invDir.Z;
            t2 = (bmax.Z - ray.origin.Z) * ray.invDir.Z;
            tmin = XMath.Max(tmin, XMath.Min(t1, t2));
            tmax = XMath.Min(tmax, XMath.Max(t1, t2));

            return tmax >= XMath.Max(tmin, tMin) && tmin <= tMax;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Reflect(Float3 I, Float3 N)
        {
            return I - N * (2f * Float3.Dot(I, N));
        }

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
        private static Ray MakeRayOffset(Ray wray, float t, Float3 n, Float3 dir, float eps)
        {
            Float3 o = wray.origin + wray.dir * t + n * eps;
            Float3 d = Float3.Normalize(dir);
            Float3 inv = new Float3(1f / (d.X != 0f ? d.X : 1e-8f), 1f / (d.Y != 0f ? d.Y : 1e-8f), 1f / (d.Z != 0f ? d.Z : 1e-8f));
            return new Ray { origin = o, dir = d, invDir = inv };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray MakeRayOffsetAlongDir(Ray wray, float t, Float3 dir, float eps)
        {
            Float3 d = Float3.Normalize(dir);
            Float3 o = wray.origin + wray.dir * t + d * eps;
            Float3 inv = new Float3(
                1f / (d.X != 0f ? d.X : 1e-8f),
                1f / (d.Y != 0f ? d.Y : 1e-8f),
                1f / (d.Z != 0f ? d.Z : 1e-8f));
            return new Ray { origin = o, dir = d, invDir = inv };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SkyColor(Float3 dir)
        {
            float tbg = 0.5f * (dir.Y + 1.0f);
            Float3 c1 = new Float3(1f, 1f, 1f);
            Float3 c2 = new Float3(0.5f, 0.7f, 1.0f);
            return c1 * (1f - tbg) + c2 * tbg;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToByte(float x)
        {
            float c = XMath.Min(1f, XMath.Max(0f, x));
            return (int)(255.99f * c);
        }
    }
}
