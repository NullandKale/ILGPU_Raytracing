using System.Runtime.CompilerServices;

namespace ILGPU_Raytracing.Engine
{
    public struct Ray
    {
        public Float3 origin;
        public Float3 dir;
        public Float3 invDir;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Ray GenerateRay(Camera cam, float u, float v)
        {
            Float3 dir = Float3.Normalize(cam.lowerLeft + cam.horizontal * u + cam.vertical * v - cam.origin);
            return new Ray { origin = cam.origin, dir = dir, invDir = new Float3(1f / (dir.X != 0f ? dir.X : 1e-8f), 1f / (dir.Y != 0f ? dir.Y : 1e-8f), 1f / (dir.Z != 0f ? dir.Z : 1e-8f)) };
        }
    }
}
