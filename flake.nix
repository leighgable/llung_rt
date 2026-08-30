{
  description = "Rust ONNX framework environment for iGPU/CPU/NPU";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-build-systems,
      pyproject-nix,
      rust-overlay,
      uv2nix,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./python; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          python = pkgs.python314;
          hacks = pkgs.callPackage pyproject-nix.build.hacks { };

          projectOverlay = final: prev: {
            torch =
              (hacks.nixpkgsPrebuilt {
                from = python.pkgs.torchWithRocm;
                prev = prev.torch;
              }).overrideAttrs
                (old: {
                  passthru = (old.passthru or { }) // {
                    dependencies = lib.filterAttrs (
                      name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                    ) (old.passthru.dependencies or { });
                  };
                });
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              projectOverlay
            ]
          )
      );
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          }; # nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "llung-rt-dev-env" workspace.deps.all;
          rocmEnv = pkgs.symlinkJoin {
            name = "rocm-combined";
            paths = with pkgs.rocmPackages; [
              rocblas
              hipblas
              clr # Contains hipcc and the HIP runtime
              clr.icd # Contains the OpenCL ICD
              rocminfo # Useful for verifying ROCm detection
            ];
          };
        in
        {
          default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            buildInputs = [
              (pkgs.rust-bin.selectLatestNightlyWith (
                toolchain:
                toolchain.default.override {
                  extensions = [
                    "rust-src"
                    "rustfmt"
                    "rustc"
                    "clippy"
                  ];
                }
              ))
              pkgs.openssl
              pkgs.pkg-config
              rocmEnv
              pkgs.vulkan-tools
              pkgs.vulkan-loader
              pkgs.vulkan-headers
              pkgs.clinfo # Useful for verifying GPU detection
              pkgs.ocl-icd # OpenCL loader
              pkgs.perf
              pkgs.rust-analyzer
            ];
            packages = [
              virtualenv
              pkgs.uv
              # pkgs.nodejs
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
                unset PYTHONPATH
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              # Allow pure-Rust WGPU / Vulkan runtimes to find system laptop drivers
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/wrappers/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "llung-rt-env" workspace.deps.default;
      });
    };
}

#             # Allow pure-Rust WGPU / Vulkan runtimes to find system laptop drivers
#             export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/wrappers/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"
#           '';
#         };
#       }
#     );
# }
# export HIP_PATH=${pkgs.rocmPackages.clr}
# # Tell the OpenCL loader where to find the AMD ICD
# export OCL_ICD_VENDORS=${pkgs.rocmPackages.clr.icd}/etc/OpenCL/vendors
# # Ensure libraries can find OpenCL and ROCm at runtime
# export LD_LIBRARY_PATH=${pkgs.ocl-icd}/lib:${rocmEnv}/lib:$LD_LIBRARY_PATH
# export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
