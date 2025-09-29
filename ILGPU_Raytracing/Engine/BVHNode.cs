namespace ILGPU_Raytracing.Engine
{
    public struct BVHNode
    {
        public Float3 boundsMin;
        public Float3 boundsMax;
        public int left;
        public int right;
        public int first;
        public int count;
        public int skipIndex;
    }
}
