{ pkgs ? import <nixpkgs> {} }:
let python =
      let
        packageOverrides = self:
          super: {
            opencv4 = super.opencv4.override {
              enableGtk2 = true;
              gtk2 = pkgs.gtk2;
              #enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
              #ffmpeg_3 = pkgs.ffmpeg-full;
            };
          };
      in
        pkgs.python311Full.override {inherit packageOverrides; self = python;};
in
pkgs.mkShell {
  packages = [
    #pkgs.python311Full
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.pip

    pkgs.python311Packages.fastapi

    pkgs.python311Packages.tensorflow
    pkgs.python311Packages.keras
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pydantic
    pkgs.python311Packages.uvicorn
    pkgs.python311Packages.bson
    pkgs.python311Packages.websockets
    pkgs.python311Packages.python-multipart
    pkgs.gcc
    pkgs.glfw


    # all pytorch stuff from colab notebook
    # also need to have mediapipe inside venv
    pkgs.python311Packages.torch
    pkgs.python311Packages.torchvision
    pkgs.python311Packages.pandas
    pkgs.python311Packages.pyarrow
    pkgs.python311Packages.fastparquet
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.seaborn
    pkgs.python311Packages.tqdm
    pkgs.python311Packages.matplotlib
    # pkgs.python311Packages.opencv4
    (python.withPackages(ps: with ps; [
      opencv4
    ]))
    pkgs.python311Packages.protobuf
    pkgs.python311Packages.av
    pkgs.zlib
    pkgs.glib
    pkgs.wget
    pkgs.git
    pkgs.ffmpeg

    # For llm inference
    # pkgs.python311Packages.groq from pip

    # For tts
    pkgs.python311Packages.soundfile
    pkgs.python311Packages.munch
    # pkgs.python311Packages.espeakng-phonemizer
    pkgs.python311Packages.phonemizer
    pkgs.python311Packages.pydub
    pkgs.python311Packages.transformers
    # also needs espeakng-loader inside venv
    pkgs.espeak


  ];
  propagatedBuildInputs = [
    #pkgs.libffi
    #pkgs.libGL
  ];
  buildInputs = [
    #pkgs.libffi
  ];
  inputsFrom = [
    # pkgs.gcc
    #pkgs.glibc
    #pkgs.gcc-unwrapped
  ];

  # inputsFrom = [ pkgs.hello pkgs.gnutar ];
  

  shellHook = ''
  export LIBSPEAK_NG_LIB=${pkgs.espeak}/lib
  export LIBSPEAK_NG_DATA=${pkgs.espeak}/share/espeak-ng-data

  # Provides with lib std c++ so 
  export LD_LIBRARY_PATH=${pkgs.libgcc.lib}/lib:$LD_LIBRARY_PATH 
  #export LD_LIBRARY_PATH=${pkgs.gcc}/lib:$LD_LIBRARY_PATH
  #export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
  #export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
  #export LD_LIBRARY_PATH=${pkgs.glib}/lib:$LD_LIBRARY_PATH
  #export LD_LIBRARY_PATH=${pkgs.wayland}/lib:$LD_LIBRARY_PATH
  unset WAYLAND_DISPLAY
  alias py=python
  export TF_CPP_MIN_LOG_LEVEL=1
  export MPLBACKEND=TkAgg

  export VIRTUAL_ENV='./python-venv'
  export PATH="$VIRTUAL_ENV/bin:$PATH"

  # unset PYTHONHOME if set
  if ! [ -z "$\{PYTHONHOME+_\}" ] ; then
      _OLD_VIRTUAL_PYTHONHOME="$PYTHONHOME"
      unset PYTHONHOME
  fi
  export PYTHONPATH=${pkgs.python311Packages.protobuf}/lib/python3.11/site-packages:$PYTHONPATH
  '';
}
