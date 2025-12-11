{
  description = "Rasterizer";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=release-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nixgl.overlays.default ];
        };

        nix_pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ nixgl.overlay ];
        };
        
        devTools = with pkgs; [
        ];

        nativeBuildInputs = with pkgs; [
          git
        ];

        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          tqdm
          numpy
          pillow
          matplotlib
          (ps.buildPythonPackage rec {
            pname = "fxpmath";
            version = "0.4.9";
            format = "pyproject";
            src = ps.fetchPypi {
              inherit pname version;
              sha256 = "456a0ae8960c9de2bd7a9518bbc9d62a22ad1f3c51b8c31e4000aeaf4f898b75";
            };
            nativeBuildInputs = [ ps.setuptools ps.wheel ps.build ];
            propagatedBuildInputs = [ ps.numpy ];
          })
          ps.pip
        ]);

        x11Packages = with pkgs; [
          libGL
          libGLU
          xorg.xhost
          xorg.xauth
          xorg.libX11
          xorg.libXrandr
          xorg.libXi
          xorg.libXcursor
          xorg.libXinerama
          xorg.libXrender
          xorg.libXfixes
          xorg.libXdamage
          xorg.libXcomposite
          xorg.libXt
          xorg.libSM
          xorg.libICE
        ];
      in
      rec {
        devShells.default = pkgs.mkShell {
          name = "Rasterizer";
          packages = devTools 
            ++ nativeBuildInputs 
            ++ [ pythonEnv (nixgl.packages.${system}.nixGLIntel) ]
            ++ x11Packages;
        };
      });
}
