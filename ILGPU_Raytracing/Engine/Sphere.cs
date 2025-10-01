namespace ILGPU_Raytracing.Engine
{
    public struct Sphere
    {
        public const int SHADING_LAMBERT = 0;
        public const int SHADING_MIRROR = 1;
        public const int SHADING_GLASS = 2;

        public Float3 center;
        public float radius;
        public Float3 albedo;
        public MaterialRecord material;
        public int shading;            // 0=lambert, 1=mirror, 2=glass
        public float ior;              // used for glass; typical 1.3-1.7
    }
}
