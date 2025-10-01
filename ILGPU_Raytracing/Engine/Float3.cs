using ILGPU.Algorithms;
using System.Runtime.CompilerServices;

namespace ILGPU_Raytracing.Engine
{
    public struct Float3
    {
        public float X;
        public float Y;
        public float Z;

        public Float3(float x, float y, float z)
        {
            X = x; Y = y; Z = z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator +(Float3 a, Float3 b)
        {
            return new Float3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator -(Float3 a, Float3 b)
        {
            return new Float3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator *(Float3 a, float s)
        {
            return new Float3(a.X * s, a.Y * s, a.Z * s);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator *(float s, Float3 a)
        {
            return new Float3(a.X * s, a.Y * s, a.Z * s);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator *(Float3 a, Float3 b)
        {
            return new Float3(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator /(Float3 a, float s)
        {
            float inv = 1f / s;
            return new Float3(a.X * inv, a.Y * inv, a.Z * inv);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator /(Float3 a, Float3 b)
        {
            return new Float3(a.X / b.X, a.Y / b.Y, a.Z / b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 operator -(Float3 v)
        {
            return new Float3(-v.X, -v.Y, -v.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Min(Float3 a, Float3 b)
        {
            return new Float3(XMath.Min(a.X, b.X), XMath.Min(a.Y, b.Y), XMath.Min(a.Z, b.Z));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Max(Float3 a, Float3 b)
        {
            return new Float3(XMath.Max(a.X, b.X), XMath.Max(a.Y, b.Y), XMath.Max(a.Z, b.Z));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Cross(Float3 a, Float3 b)
        {
            return new Float3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot(Float3 a, Float3 b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Normalize(Float3 v)
        {
            float inv = XMath.Rsqrt(XMath.Max(1e-20f, v.X * v.X + v.Y * v.Y + v.Z * v.Z));
            return new Float3(v.X * inv, v.Y * inv, v.Z * inv);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Saturate(Float3 c)
        {
            return new Float3(XMath.Min(1f, XMath.Max(0f, c.X)), XMath.Min(1f, XMath.Max(0f, c.Y)), XMath.Min(1f, XMath.Max(0f, c.Z)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Length(Float3 v)
        {
            return XMath.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float3 Center(Float3 a, Float3 b)
        {
            return new Float3(0.5f * (a.X + b.X), 0.5f * (a.Y + b.Y), 0.5f * (a.Z + b.Z));
        }
    }
}
