// File: Engine/Camera/CameraController.cs
using System;
using ILGPU.Algorithms;
using ILGPU.Runtime.Cuda;
using ILGPU;
using System.Runtime.CompilerServices;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace ILGPU_Raytracing.Engine
{
    public interface CameraController
    {
        void Update(ref Camera camera, float dtSeconds);
    }

    public sealed class FlyCameraController : CameraController, IDisposable
    {
        private readonly RTWindow _window;
        private float _fovDegrees = 60f;
        private float _mouseSensitivityDegPerPixel = 0.08f;
        private float _baseSpeed = 3.0f;
        private float _fastMultiplier = 4.0f;
        private float _slowMultiplier = 0.25f;
        private float _pendingScrollY;

        public FlyCameraController(RTWindow window)
        {
            _window = window ?? throw new ArgumentNullException(nameof(window));
            _window.MouseWheel += OnMouseWheel;
        }

        public void Update(ref Camera camera, float dtSeconds)
        {
            // Single source of truth: the window
            bool captured = _window.IsMouseCaptured;

            if (captured)
            {
                var md = _window.ConsumeMouseDelta();
                if (md.X != 0f || md.Y != 0f)
                    camera.OnMouseLook(md.X, md.Y, _mouseSensitivityDegPerPixel);
            }

            float speed = _baseSpeed;
            if (_window.IsKeyDown(Keys.LeftShift) || _window.IsKeyDown(Keys.RightShift)) speed *= _fastMultiplier;
            if (_window.IsKeyDown(Keys.LeftControl) || _window.IsKeyDown(Keys.RightControl)) speed *= _slowMultiplier;

            bool w = _window.IsKeyDown(Keys.W);
            bool a = _window.IsKeyDown(Keys.A);
            bool s = _window.IsKeyDown(Keys.S);
            bool d = _window.IsKeyDown(Keys.D);

            // Remap vertical to avoid 'E' conflict with capture toggle
            bool down = _window.IsKeyDown(Keys.C);
            bool up = _window.IsKeyDown(Keys.Space);

            camera.OnKeyboardFly(w, a, s, d, down, up, dtSeconds, speed);

            if (_pendingScrollY != 0f)
            {
                _fovDegrees = Clamp(_fovDegrees - _pendingScrollY * 5f, 20f, 100f);
                _pendingScrollY = 0f;
            }

            float aspect = (float)_window.Size.X / Math.Max(1, _window.Size.Y);
            camera.SetFov(_fovDegrees, aspect);
        }

        private void OnMouseWheel(Vector2 offset) => _pendingScrollY += offset.Y;

        private static float Clamp(float x, float lo, float hi) => x < lo ? lo : (x > hi ? hi : x);

        public void Dispose()
        {
            _window.MouseWheel -= OnMouseWheel;
        }
    }
}

