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
            pname = "fixedpoint";
            version = "1.0.1";
            format = "pyproject";
            src = ps.fetchPypi {
              inherit pname version;
              sha256 = "48339fbb3adb47f03ab325debc3f37166d6632f92dfd29a7e200c31e4054f19e";
            };
            postPatch = ''
              mkdir -p docs/source
              touch docs/source/long_description.rst
              sed -i '/nose>=1.3.7/d' setup.cfg
              sed -i '/test_suite = nose.collector/d' setup.cfg
            '';
            nativeBuildInputs = [ ps.setuptools ps.wheel ps.build ];
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
