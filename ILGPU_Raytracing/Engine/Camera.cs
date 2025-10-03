using ILGPU.Algorithms;

namespace ILGPU_Raytracing.Engine
{
    public struct Camera
    {
        public Float3 origin;
        public Float3 lowerLeft;
        public Float3 horizontal;
        public Float3 vertical;

        // NEW: cached basis + lens params used for reprojection
        public Float3 forward;       // view forward (unit)
        public Float3 right;         // view right (unit)
        public Float3 up;            // view up (unit)
        public float aspect;         // width/height
        public float fovYRadians;    // vertical FoV in radians

        public static Camera CreateCamera(int width, int height, float fovDegrees)
        {
            float aspect = (float)width / (float)XMath.Max(1, height);
            float theta = fovDegrees * (XMath.PI / 180f);
            float halfHeight = XMath.Tan(0.5f * theta);
            float halfWidth = aspect * halfHeight;

            Float3 origin = new Float3(0f, 1f, 3f);
            Float3 lookAt = new Float3(0f, 0.5f, 0f);
            Float3 upHint = new Float3(0f, 1f, 0f);

            Float3 w = Normalize(origin - lookAt);
            Float3 u = Normalize(Cross(upHint, w));
            Float3 v = Cross(w, u);

            Float3 lowerLeft = origin - u * halfWidth - v * halfHeight - w;
            Float3 horizontal = u * (2f * halfWidth);
            Float3 vertical = v * (2f * halfHeight);

            Camera cam = new Camera
            {
                origin = origin,
                lowerLeft = lowerLeft,
                horizontal = horizontal,
                vertical = vertical
            };
            cam.UpdateDerived(aspect, theta); // set basis/fov/aspect
            return cam;
        }

        // Mouse look: keep fov/aspect unchanged
        public void OnMouseLook(float deltaX, float deltaY, float sensitivityDegPerPixel = 0.08f)
        {
            float yawDeg = deltaX * sensitivityDegPerPixel;
            float pitchDeg = -deltaY * sensitivityDegPerPixel;
            RotateYawPitch(yawDeg, pitchDeg);
        }

        public void OnKeyboardFly(
            bool keyW, bool keyA, bool keyS, bool keyD, bool keyQ, bool keyE,
            float dtSeconds, float moveSpeed = 3.0f)
        {
            Float3 forward = Normalize((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            Float3 up = Normalize(vertical);
            Float3 right = Normalize(Cross(forward, up));
            Float3 worldUp = new Float3(0f, 1f, 0f);

            Float3 fwdHoriz = forward - worldUp * Dot(forward, worldUp);
            float fwdLen2 = Dot(fwdHoriz, fwdHoriz);
            fwdHoriz = (fwdLen2 > 1e-12f) ? fwdHoriz * XMath.Rsqrt(fwdLen2) : right;

            Float3 move = default;
            if (keyA) move -= right;
            if (keyD) move += right;
            if (keyQ) move -= worldUp;
            if (keyE) move += worldUp;
            if (keyW) move += fwdHoriz;
            if (keyS) move -= fwdHoriz;

            float m2 = Dot(move, move);
            if (m2 > 1e-12f)
            {
                move = move * XMath.Rsqrt(m2);
                Translate(move * (moveSpeed * dtSeconds));
            }
        }

        public Camera(Float3 origin, Float3 lowerLeft, Float3 horizontal, Float3 vertical)
        {
            this.origin = origin;
            this.lowerLeft = lowerLeft;
            this.horizontal = horizontal;
            this.vertical = vertical;
            // Fill derived with defaults (caller should call UpdateDerived)
            forward = new Float3(0, 0, -1);
            right = new Float3(1, 0, 0);
            up = new Float3(0, 1, 0);
            aspect = 1f;
            fovYRadians = XMath.PI / 3f;
        }

        public Camera(Float3 origin, Float3 lookAt, Float3 up, float vfovDegrees, float aspect, float focusDist = 1f)
        {
            float theta = DegToRad(vfovDegrees);
            float halfHeight = XMath.Tan(0.5f * theta);
            float halfWidth = aspect * halfHeight;

            Float3 forward = Normalize(lookAt - origin);
            OrthoBasis(forward, up, out Float3 u, out Float3 v, out Float3 w);

            this.origin = origin;
            this.horizontal = u * (2f * halfWidth);
            this.vertical = v * (2f * halfHeight);
            this.lowerLeft = origin - u * halfWidth - v * halfHeight + forward * focusDist;

            this.forward = Normalize((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            this.right = Normalize(Cross(this.forward, v));
            this.up = Normalize(v);
            this.aspect = aspect;
            this.fovYRadians = theta;
        }

        public void Translate(Float3 delta)
        {
            origin += delta;
            lowerLeft += delta;
            UpdateDerived(aspect, fovYRadians);
        }

        public void SetFov(float vfovDegrees, float aspect)
        {
            float focusDist = Length((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            Float3 forward = Normalize((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            Float3 up = Normalize(vertical);

            float theta = DegToRad(vfovDegrees);
            float halfHeight = XMath.Tan(0.5f * theta);
            float halfWidth = aspect * halfHeight;

            OrthoBasis(forward, up, out Float3 u, out Float3 v, out Float3 w);

            horizontal = u * (2f * halfWidth);
            vertical = v * (2f * halfHeight);
            lowerLeft = origin - u * halfWidth - v * halfHeight + forward * focusDist;

            UpdateDerived(aspect, theta);
        }

        public void RotateYawPitch(float yawDegrees, float pitchDegrees)
        {
            float halfWidth = 0.5f * Length(horizontal);
            float halfHeight = 0.5f * Length(vertical);
            float focusDist = Length((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);

            Float3 forward = Normalize((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            Float3 upVec = Normalize(vertical);
            Float3 rightVec = Normalize(Cross(forward, upVec));
            Float3 worldUp = new Float3(0f, 1f, 0f);

            float yaw = DegToRad(yawDegrees);
            float pitch = DegToRad(pitchDegrees);

            if (Abs(Dot(forward, worldUp)) > 0.999f)
                worldUp = Normalize(Cross(rightVec, forward));

            forward = RotateAroundAxis(forward, worldUp, yaw);
            upVec = RotateAroundAxis(upVec, worldUp, yaw);
            rightVec = Normalize(Cross(forward, upVec));
            upVec = Normalize(Cross(rightVec, forward));

            forward = RotateAroundAxis(forward, rightVec, pitch);
            upVec = Normalize(Cross(rightVec, forward));

            OrthoBasis(forward, upVec, out Float3 u, out Float3 v, out Float3 w);

            horizontal = u * (2f * halfWidth);
            vertical = v * (2f * halfHeight);
            lowerLeft = origin - u * halfWidth - v * halfHeight + forward * focusDist;

            // keep the same fov/aspect
            UpdateDerived(aspect, fovYRadians);
        }

        // --- helpers ---

        private void UpdateDerived(float aspectIn, float fovYRadIn)
        {
            forward = Normalize((lowerLeft + horizontal * 0.5f + vertical * 0.5f) - origin);
            up = Normalize(vertical);
            right = Normalize(Cross(forward, up));
            aspect = aspectIn;
            fovYRadians = fovYRadIn;
        }

        private static void OrthoBasis(Float3 forward, Float3 upHint, out Float3 u, out Float3 v, out Float3 w)
        {
            Float3 f = Normalize(forward);
            Float3 up = upHint;
            if (Abs(Dot(f, up)) > 0.999f)
            {
                up = new Float3(0f, 1f, 0f);
                if (Abs(Dot(f, up)) > 0.999f) up = new Float3(1f, 0f, 0f);
            }
            u = Normalize(Cross(f, up));
            v = Normalize(Cross(u, f));
            w = new Float3(-f.X, -f.Y, -f.Z);
        }

        private static Float3 RotateAroundAxis(Float3 v, Float3 axis, float angleRad)
        {
            Float3 a = Normalize(axis);
            float c = XMath.Cos(angleRad);
            float s = XMath.Sin(angleRad);
            Float3 term1 = v * c;
            Float3 term2 = Cross(a, v) * s;
            Float3 term3 = a * (Dot(a, v) * (1f - c));
            return term1 + term2 + term3;
        }

        private static float DegToRad(float d) => d * (XMath.PI / 180f);
        private static float Length(Float3 v) => XMath.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        private static Float3 Normalize(Float3 v)
        {
            float len2 = v.X * v.X + v.Y * v.Y + v.Z * v.Z;
            float inv = XMath.Rsqrt(XMath.Max(1e-20f, len2));
            return new Float3(v.X * inv, v.Y * inv, v.Z * inv);
        }
        private static Float3 Cross(Float3 a, Float3 b) =>
            new Float3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
        private static float Dot(Float3 a, Float3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        private static float Abs(float x) => x < 0f ? -x : x;
    }
}
