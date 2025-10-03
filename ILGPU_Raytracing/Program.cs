using System;
using System;
using System.Collections.Generic;
using ILGPU_Raytracing.Engine;
using OpenTK.Mathematics;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace ILGPU_Raytracing
{
    public class Program
    {
        static void Main(string[] args)
        {
            using (var window = new RTWindow(1280, 720, ""))
            {
                window.Run();
            }
        }
    }
}