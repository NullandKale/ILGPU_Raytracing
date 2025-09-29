namespace ILGPU_Raytracing.Engine
{
    public struct Affine3x4
    {
        public float m00; public float m01; public float m02; public float m03;
        public float m10; public float m11; public float m12; public float m13;
        public float m20; public float m21; public float m22; public float m23;

        public static Affine3x4 Identity()
        {
            Affine3x4 a = default;
            a.m00 = 1f; a.m11 = 1f; a.m22 = 1f;
            return a;
        }
    }
}
