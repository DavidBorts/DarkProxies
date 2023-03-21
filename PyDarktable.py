import math
import numpy as np
import sys
import os
import struct
import subprocess
import tempfile
import rawpy
from dataclasses import dataclass, field, fields
import Constants as c

# Point me to darktable-cli for 3.8.
#_DARKTABLE_CLI = "/Applications/darktable.app/Contents/MacOS/darktable-cli"
_DARKTABLE_CLI = c.DARKTABLE_PATH

_FMT_STR = '''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
    xmlns:darktable="http://darktable.sf.net/"
   exif:DateTimeOriginal="2020:08:27 10:09:35"
   xmp:Rating="1"
   darktable:import_timestamp="1651274350"
   darktable:change_timestamp="-1"
   darktable:export_timestamp="-1"
   darktable:print_timestamp="-1"
   darktable:xmp_version="4"
   darktable:raw_params="0"
   darktable:auto_presets_applied="1"
   darktable:history_end="1000"
   darktable:iop_order_version="2">
   <darktable:history>
    <rdf:Seq>
     <rdf:li
      darktable:num="0"
      darktable:operation="rawprepare"
      darktable:enabled="1"
      darktable:modversion="1"
      darktable:params="{raw_prepare_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="1"
      darktable:operation="demosaic"
      darktable:enabled="1"
      darktable:modversion="4"
      darktable:params="0000000000000000000000000500000001000000cdcc4c3e"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="2"
      darktable:operation="colorin"
      darktable:enabled="1"
      darktable:modversion="7"
      darktable:params="gz48eJzjYhgFowABWAbaAaNgwAEANOwADw=="
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="3"
      darktable:operation="colorout"
      darktable:enabled="1"
      darktable:modversion="5"
      darktable:params="gz35eJxjZBgFo4CBAQAEEAAC"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="4"
      darktable:operation="gamma"
      darktable:enabled="1"
      darktable:modversion="1"
      darktable:params="gz35eJxjZBgFo4CB"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="5"
      darktable:operation="temperature"
      darktable:enabled="{enable_temperature}"
      darktable:modversion="3"
      darktable:params="{temperature_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="6"
      darktable:operation="highlights"
      darktable:enabled="{enable_highlights}"
      darktable:modversion="2"
      darktable:params="{highlights_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz13eJxjYGBgYARiCQYYOOHEgAZY0QVwggZ7CB6pfNoAAFDAGQk="/>
     <rdf:li
      darktable:num="7"
      darktable:operation="sharpen"
      darktable:enabled="{enable_sharpen}"
      darktable:modversion="1"
      darktable:params="{sharpen_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz13eJxjYGBgYAJiCQYYOOHEgAZY0QVwggZ7CB6pfNoAAFJgGQo="/>
     <rdf:li
      darktable:num="8"
      darktable:operation="filmicrgb"
      darktable:enabled="{enable_filmicrgb}"
      darktable:modversion="5"
      darktable:params="{filmicrgb_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz10eJxjYGBgYAFiCQYYOOHEgAZY0QVwggZ7CB6pfOygYtaVAyCMi08IAAB/xiOk"/>
     <rdf:li
      darktable:num="9"
      darktable:operation="exposure"
      darktable:enabled="{enable_exposure}"
      darktable:modversion="6"
      darktable:params="{exposure_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz10eJxjYGBgYAFiCQYYOOHEgAZY0QVwggZ7CB6pfOygYtaVAyCMi08IAAB/xiOk"/>
     <rdf:li
      darktable:num="10"
      darktable:operation="flip"
      darktable:enabled="1"
      darktable:modversion="2"
      darktable:params="ffffffff"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz14eJxjYIAACQYYOOHEgAZY0QVwggZ7CB6pfNoAAE8gGQg="/>
     <rdf:li
      darktable:num="11"
      darktable:operation="colorbalancergb"
      darktable:enabled="{enable_colorbalancergb}"
      darktable:modversion="4"
      darktable:params="{colorbalancergb_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="000000000400000018000000000000000000c84200000000000000000000000000000000050000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f0000000000000000000000000000000000000000000000000000000000000000789ad4c0789ad4c00000000000000000789ad4c0789ad4c000000000000000000000000000000000000000000000000000000000000000000000000000000000"/>
	 <rdf:li
      darktable:num="12"
      darktable:operation="hazeremoval"
      darktable:enabled="{enable_hazeremoval}"
      darktable:modversion="1"
      darktable:params="{hazeremoval_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="000000000400000018000000000000000000c84200000000000000000000000000000000050000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f0000000000000000000000000000000000000000000000000000000000000000789ad4c0789ad4c00000000000000000789ad4c0789ad4c000000000000000000000000000000000000000000000000000000000000000000000000000000000"/>
      <rdf:li
      darktable:num="13"
      darktable:operation="denoiseprofile"
      darktable:enabled="{enable_denoiseprofile}"
      darktable:modversion="11"
      darktable:params="{denoiseprofile_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="000000000400000018000000000000000000c84200000000000000000000000000000000050000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f0000000000000000000000000000000000000000000000000000000000000000789ad4c0789ad4c00000000000000000789ad4c0789ad4c000000000000000000000000000000000000000000000000000000000000000000000000000000000"/>
      <rdf:li
      darktable:num="14"
      darktable:operation="lowpass"
      darktable:enabled="{enable_lowpass}"
      darktable:modversion="4"
      darktable:params="{lowpass_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="000000000200000018000000000000000000c84200000000000000000000000000000000050000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"/>
      <rdf:li
      darktable:num="15"
      darktable:operation="censorize"
      darktable:enabled="{enable_censorize}"
      darktable:modversion="1"
      darktable:params="{censorize_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="000000000400000018000000000000000000c84200000000000000000000000000000000050000000000000000000000000000000000000000000000000000000000000000000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f00000000000000000000803f0000803f0000000000000000000000000000000000000000000000000000000000000000789ad4c0789ad4c00000000000000000789ad4c0789ad4c000000000000000000000000000000000000000000000000000000000000000000000000000000000"/>  
    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


def to_hex_string(x):
    if type(x) == bytes:
        # Convert each byte into two hex characters and concat into a string.
        return ''.join([format(byte, '02x') for byte in x])
    elif type(x) == float:
        return to_hex_string(struct.pack('f', x))
    elif type(x) == int or type(x) == bool:
        # Pack bool as int (4 bytes).
        return to_hex_string(struct.pack('i', x))
    elif type(x) == np.uint16:
        return to_hex_string(struct.pack('H', x))
    elif type(x) == list:
        return ''.join([to_hex_string(y) for y in x])
    else:
        raise ValueError("Unsupported type: %s" % type(x))

@dataclass
class RawPrepareParams:
    # Parse these out of Default Crop Origin using exiftool
    x: int = 0
    y: int = 0
    # Parse these out of Default Crop Size using exiftool
    width: int = 0
    height: int = 0
    # Parse this out as "Black Level" and "Black Level Repeat Dim".
    # If repeat dim is (1, 1)), then it's a single value and just repeat it 4 times.
    # If it's (2, 2), then repeat it's 4 values.
    # Use make_black_levels() to get the type right.
    black_levels: list[np.uint16] = field(
        default_factory=lambda: [np.uint16(0)] * 4)
    # Actually uint16 as well, but because of struct alignment, it's packed as an int.
    white_point: int = 0

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])


def make_black_levels(bl_0, bl_1, bl_2, bl_3):
    return [np.uint16(bl_0), np.uint16(bl_1), np.uint16(bl_2), np.uint16(bl_3)]


@dataclass
class ColorBalanceRGBParams:
    # v1 params: 24 deprecated floats but they seem to have some effect in the code, ugh.
    shadows_Y: float = 0.0
    shadows_C: float = 0.0
    shadows_H: float = 0.0
    midtones_Y: float = 0.0
    midtones_C: float = 0.0
    midtones_H: float = 0.0
    highlights_Y: float = 0.0
    highlights_C: float = 0.0
    highlights_H: float = 0.0
    global_Y: float = 0.0
    global_C: float = 0.0
    global_H: float = 0.0
    shadows_weight: float = 1.0
    white_fulcrum: float = 0.0
    highlights_weight: float = 1.0
    chroma_shadows: float = 0.0
    chroma_highlights: float = 0.0
    chroma_global: float = 0.0
    chroma_midtones: float = 0.0
    saturation_global: float = 0.0
    saturation_highlights: float = 0.0
    saturation_midtones: float = 0.0
    saturation_shadows: float = 0.0
    hue_angle: float = 0.0

    # v2 params: 4 deprecated floats
    brilliance_global: float = 0.0
    brilliance_highlights: float = 0.0
    brilliance_midtones: float = 0.0
    brilliance_shadows: float = 0.0

    # v3 params: 1 deprecated param
    mask_grey_fulcrum: float = 0.1845

    # v4 params (current)
    vibrance: float = 0.0  # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0 $DESCRIPTION: "global vibrance"
    grey_fulcrum: float = 0.1845  # $MIN:  0.0 $MAX: 1.0 $DEFAULT: 0.1845 $DESCRIPTION: "contrast gray fulcrum"
    contrast: float = 0.0  # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0. $DESCRIPTION: "contrast"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

# dt_iop_exposure_params_t
@dataclass
class ExposureParams:
    # In C, they are enums:
    # MANUAL = 0     "manual"
    # DEFLICKER = 1  "automatic"
    mode: int = 0

    black: float = 0.0
    exposure: float = 0.0
    deflicker_percentile: float = 50.0
    deflicker_target_level: float = -4.0

    compensate_exposure_bias: bool = False

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])


@dataclass
class FilmicRGBParams:
    grey_point_source: float = 18.45
    black_point_source: float = -7.75
    white_point_source: float = 4.400000095367432
    reconstruct_threshold: float = 3.0
    reconstruct_feather: float = 3.0
    reconstruct_bloom_vs_details: float = 100.0
    reconstruct_grey_vs_color: float = 100.0
    reconstruct_structure_vs_texture: float = 0.0
    security_factor: float = 0.0
    grey_point_target: float = 18.45
    black_point_target: float = 0.01517634
    white_point_target: float = 100.0
    output_power: float = 3.75882887840271
    latitude: float = 50.0
    contrast: float = 1.1
    saturation: float = 0.0
    balance: float = 0.0
    noise_level: float = 0.2

    # enum dt_iop_filmicrgb_methods_type_t
    preserve_color: int = 3  # "preserve chrominance"
    # enum dt_iop_filmicrgb_colorscience_type_t
    # DT_FILMIC_COLORSCIENCE_V3 = 2
    version: int = 2  # "color science"
    auto_hardness: bool = True  # "auto adjust hardness"
    custom_grey: bool = False  # "use custom middle-gray values"
    high_quality_reconstruction: int = 1  # "iterations of high-quality reconstruction"
    # enum dt_iop_filmic_noise_distribution_t
    # DT_NOISE_GAUSSIAN = 1
    noise_distribution: int = 1  # "type of noise"

    # dt_iop_filmicrgb_curve_type_t
    # DT_FILMIC_CURVE_RATIONAL = 2
    shadows: int = 2  #  "contrast in shadows"
    highlights: int = 2  #  "contrast in highlights"

    compensate_icc_black: bool = False  # "compensate output ICC profile black point"

    # enum dt_iop_filmicrgb_spline_version_type_t
    # DT_FILMIC_SPLINE_VERSION_V3 = 2
    spline_version: int = 2  # "spline handling"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])


# dt_iop_highlights_params_t
#TODO: Update to reflected updated params in Darktable highlights.c source code
@dataclass
class HighlightsParams:
    # In C, they are enums:
    # CLIP = 0,
    # LCH = 1,
    # INPAINT = 2 ("reconstruct color").
    mode: int = 0

    # In C, the comments say they're unused.
    blendL: float = 1.0
    blendC: float = 0.0
    blendH: float = 0.0

    # Clipping threshold.
    clip: float = 1.0

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class SharpenParams:
    radius: float = 8.0
    amount: float = 0.5
    threshold: float = 0.5

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class HazeRemovalParams:
    strength: float = 0.2 # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.2
    distance: float = 0.2 # $MIN:  0.0 $MAX: 1.0 $DEFAULT: 0.2
    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class DenoiseProfileParams:
    radius: float = 1.0 # $MIN: 0.0 $MAX: 12.0 $DEFAULT: 1.0 $DESCRIPTION: "patch size"
    nbhood: float = 7.0 # $MIN: 1.0 $MAX: 30.0 $DEFAULT: 7.0 $DESCRIPTION: "search radius"
    strength: float = 1.0 # $MIN: 0.001 $MAX: 1000.0 $DEFAULT: 1.0 $DESCRIPTION: "noise level after equalization"
    shadows: float = 1.0 # $MIN: 0.0 $MAX: 1.8 $DEFAULT: 1.0 $DESCRIPTION: "preserve shadows"
    bias: float = 0.0 # $MIN: -1000.0 $MAX: 100.0 $DEFAULT: 0.0 $DESCRIPTION: "bias correction"
    scattering: float = 0.0 # $MIN: 0.0 $MAX: 20.0 $DEFAULT: 0.0 $DESCRIPTION: "spread the patch search zone without increasing number of patches"
    central_pixel_weight: float = 0.1 # $MIN: 0.0 $MAX: 10.0 $DEFAULT: 0.1 $DESCRIPTION: "increase central pixel's weight in patch comparison"
    overshooting: float = 1.0 # $MIN: 0.001 $MAX: 1000.0 $DEFAULT: 1.0 $DESCRIPTION: "adjust autoset parameters"

    a: list[float] = field(default_factory=lambda: [0.5] * 3) # fit for poissonian-gaussian noise per color channel.
    b: list[float] = field(default_factory=lambda: [0.5] * 3) # fit for poissonian-gaussian noise per color channel.
    # float a[3], b[3]; // fit for poissonian-gaussian noise per color channel.

    # In the original C code, they are enums:
    # MODE_NLMEANS = 0,
    # MODE_WAVELETS = 1,
    # MODE_VARIANCE = 2,
    # MODE_NLMEANS_AUTO = 3,
    # MODE_WAVELETS_AUTO = 4
    mode: int = 0 # switch between nlmeans and wavelets $DEFAULT: MODE_WAVELETS 

    x: list[list[float]] = field(default_factory=lambda: [[0.5] * 7] * 6) # values to change wavelet force by frequency $DEFAULT: 0.5
    #x: float[6][7] = [[0.5]*7]*6 # values to change wavelet force by frequency $DEFAULT: 0.5
    y: list[list[float]] = field(default_factory=lambda: [[0.5] * 7] * 6) # values to change wavelet force by frequency $DEFAULT: 0.5

    wb_adaptive_anscombe: bool = True # $DEFAULT: TRUE $DESCRIPTION: "whitebalance-adaptive transform" whether to adapt anscombe transform to wb coeffs"
    fix_anscombe_and_nlmeans_norm: bool = True # $DEFAULT: TRUE $DESCRIPTION: "fix various bugs in algorithm" backward compatibility options"
    use_new_vst: bool = True # $DEFAULT: TRUE $DESCRIPTION: "upgrade profiled transform" backward compatibility options"

    # In the original C code, they are enums:
    # MODE_RGB = 0,    $DESCRIPTION: "RGB"
    # MODE_Y0U0V0 = 1  $DESCRIPTION: "Y0U0V0"
    wavelet_color_mode: int = 1 # $DEFAULT: MODE_Y0U0V0 $DESCRIPTION: "color mode"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class LowpassParams:
    # dt_gaussian_order_t objects in the C code
    order: int = 0 # $DEFAULT: 0

    radius: float = 10.0 # $MIN: 0.1 $MAX: 500.0 $DEFAULT: 10.0
    contrast: float = 1.0 # $MIN: -3.0 $MAX: 3.0 $DEFAULT: 1.0
    brightness: float = 0.0 # $MIN: -3.0 $MAX: 3.0 $DEFAULT: 0.0
    saturation: float = 1.0 # $MIN: -3.0 $MAX: 3.0 $DEFAULT: 1.0

    # In the original C code, they are enums:
    # LOWPASS_ALGO_GAUSSIAN = 0, $DESCRIPTION: "gaussian"
    # LOWPASS_ALGO_BILATERAL = 1, $DESCRIPTION: "bilateral filter"
    lowpass_algo: int = 0 # $DEFAULT: LOWPASS_ALGO_GAUSSIAN $DESCRIPTION: "soften with"

    unbound: int = 1 #$DEFAULT: 1

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class CensorizeParams:
    radius_1: float = 0.0 # $MIN: 0.0 $MAX: 500.0 $DEFAULT: 0.0  $DESCRIPTION: "input blur radius"
    pixelate: float = 0.0 # $MIN: 0.0 $MAX: 500.0 $DEFAULT: 0.0 $DESCRIPTION: "pixellation radius"
    radius_2: float = 0.0 # $MIN: 0.0 $MAX: 500.0 $DEFAULT: 0.0  $DESCRIPTION: "output blur radius"
    noise: float = 0.0 # $MIN: 0.0 $MAX: 1.0   $DEFAULT: 0.0   $DESCRIPTION: "noise level"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class TemperatureParams:
    red: float = 1.420689582824707
    green: float = 1.0
    blue: float = 2.0
    g2: float = math.nan

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

'''
@dataclass
class ColorinParams:
    # In the original C code, they are enums:
    # DT_COLORSPACE_NONE = -1,
    # DT_COLORSPACE_FILE = 0,
    # DT_COLORSPACE_SRGB = 1,
    # DT_COLORSPACE_ADOBERGB = 2,
    # DT_COLORSPACE_LIN_REC709 = 3,
    # DT_COLORSPACE_LIN_REC2020 = 4,
    # DT_COLORSPACE_XYZ = 5,
    # DT_COLORSPACE_LAB = 6,
    # DT_COLORSPACE_INFRARED = 7,
    # DT_COLORSPACE_DISPLAY = 8,
    # DT_COLORSPACE_EMBEDDED_ICC = 9,
    # DT_COLORSPACE_EMBEDDED_MATRIX = 10,
    # DT_COLORSPACE_STANDARD_MATRIX = 11,
    # DT_COLORSPACE_ENHANCED_MATRIX = 12,
    # DT_COLORSPACE_VENDOR_MATRIX = 13,
    # DT_COLORSPACE_ALTERNATE_MATRIX = 14,
    # DT_COLORSPACE_BRG = 15,
    # DT_COLORSPACE_EXPORT = 16,
    # DT_COLORSPACE_SOFTPROOF = 17,
    # DT_COLORSPACE_WORK = 18,
    # DT_COLORSPACE_DISPLAY2 = 19,
    # DT_COLORSPACE_REC709 = 20,
    # DT_COLORSPACE_PROPHOTO_RGB = 21,
    # DT_COLORSPACE_PQ_REC2020 = 22,
    # DT_COLORSPACE_HLG_REC2020 = 23,
    # DT_COLORSPACE_PQ_P3 = 24,
    # DT_COLORSPACE_HLG_P3 = 25,
    # DT_COLORSPACE_LAST = 26
    type: int = 12 # $DEFAULT: DT_COLORSPACE_ENHANCED_MATRIX
    char filename[DT_IOP_COLOR_ICC_LEN];
    # In the original C code, they are enums:
    #DT_INTENT_PERCEPTUAL = INTENT_PERCEPTUAL,                       // 0
    #DT_INTENT_RELATIVE_COLORIMETRIC = INTENT_RELATIVE_COLORIMETRIC, // 1
    #DT_INTENT_SATURATION = INTENT_SATURATION,                       // 2
    #DT_INTENT_ABSOLUTE_COLORIMETRIC = INTENT_ABSOLUTE_COLORIMETRIC, // 3
    #DT_INTENT_LAST
    intent: int = 0 # $DEFAULT: DT_INTENT_PERCEPTUAL
    # In the original C code, they are enums:
    #DT_NORMALIZE_OFF,               //$DESCRIPTION: "off"
    #DT_NORMALIZE_SRGB,              //$DESCRIPTION: "sRGB"
    #DT_NORMALIZE_ADOBE_RGB,         //$DESCRIPTION: "Adobe RGB (compatible)"
    #DT_NORMALIZE_LINEAR_REC709_RGB, //$DESCRIPTION: "linear Rec709 RGB"
    #DT_NORMALIZE_LINEAR_REC2020_RGB //$DESCRIPTION: "linear Rec2020 RGB"
    normalize: int = 0 # $DEFAULT: DT_NORMALIZE_OFF $DESCRIPTION: "gamut clipping"
    int blue_mapping;
    // working color profile
    dt_colorspaces_color_profile_type_t type_work; // $DEFAULT: DT_COLORSPACE_LIN_REC2020
     filename_work[DT_IOP_COLOR_ICC_LEN];
'''

# Return temperature and rawprepare params for input DNG file
def read_dng_params(dng_file):
    raw_prepare_params = RawPrepareParams()
    temperature_params = TemperatureParams()
    # Using rawpy to read out params from the file
    with rawpy.imread(dng_file) as raw:
        raw_prepare_params.black_levels = [
            np.uint16(x) for x in raw.black_level_per_channel
        ]
        raw_prepare_params.white_point = int(raw.white_level)
        temperature_params.red = float(raw.camera_whitebalance[0])
        temperature_params.green = float(raw.camera_whitebalance[1])
        temperature_params.blue = float(raw.camera_whitebalance[2])
        # g2 is usually not present. DarkTable wants math.nan, rawpy returns 0
        # temperature_params.g2 = float(raw.camera_whitebalance[3])
    return raw_prepare_params, temperature_params

class functions:
    @staticmethod
    def colorbalancergb(value, params_dict, param='contrast'):
        colorbalancergb_params = ColorBalanceRGBParams()
        setattr(colorbalancergb_params, param, float(value))
        #colorbalancergb_params.contrast = float(value)
        params_dict["colorbalancergb_params"] = colorbalancergb_params
        return params_dict
    
    @staticmethod
    def highlights(value, params_dict, param='clip'):
        highlights_params = HighlightsParams()
        setattr(highlights_params, param, float(value))
        params_dict["highlights_params"] = highlights_params
        return params_dict
    
    @staticmethod
    def sharpen(value, params_dict, param='amount'):
        sharpen_params = SharpenParams()
        setattr(sharpen_params, param, float(value))
        #sharpen_params.amount = float(value)
        params_dict["sharpen_params"] = sharpen_params
        return params_dict
    
    @staticmethod
    def exposure(value, params_dict, param='exposure'):
        exposure_params = ExposureParams()
        setattr(exposure_params, param, float(value))
        #exposure_params.exposure = float(value)
        params_dict["exposure_params"] = exposure_params
        return params_dict
    
    @staticmethod
    def hazeremoval(value, params_dict, param='strength'):
        hazeremoval_params = HazeRemovalParams()
        setattr(hazeremoval_params, param, float(value))
        #hazeremoval_params.strength = float(value)
        params_dict["hazeremoval_params"] = hazeremoval_params
        return params_dict
    
    @staticmethod
    def denoiseprofile(value, params_dict, param='strength'):
        denoiseprofile_params = DenoiseProfileParams()
        setattr(denoiseprofile_params, param, float(value))
        #denoiseprofile_params.strength = float(value)
        params_dict["denoiseprofile_params"] = denoiseprofile_params
        return params_dict
    
    @staticmethod
    def lowpass(value, params_dict, param='radius'):
        lowpass_params = LowpassParams()
        setattr(lowpass_params, param, float(value))
        #lowpass_params.radius = float(value)
        params_dict["lowpass_params"] = lowpass_params
        return params_dict
    
    @staticmethod
    def censorize(value, params_dict, param='pixelate'):
        censorize_params = CensorizeParams()
        setattr(censorize_params, param, float(value))
        params_dict["censorize_params"] = censorize_params
        return params_dict

# Pipeline order is as follows, * means it can be skipped and therefore has a bool param.
# 0 rawprepare
# 1 temperature      *
# 2 highlights       *
# 3 demosaic
# 4 denoise(profiled)*
# 5 hazeremoval      *
# 6 flip             *
# 7 exposure         *
# 8 colorin
# 9 censorize        *
# 10 lowpass         *
# 11 sharpen         *
# 12 colorbalancergb *
# 13 filmicrgb       *
# 14 colorout
def get_pipe_xmp(raw_prepare_params=RawPrepareParams(),
                 temperature_params=TemperatureParams(),
                 highlights_params=HighlightsParams(),
                 denoiseprofile_params=DenoiseProfileParams(),
                 hazeremoval_params=HazeRemovalParams(),
                 exposure_params=ExposureParams(),
                 lowpass_params=LowpassParams(),
                 censorize_params=CensorizeParams(),
                 sharpen_params=SharpenParams(),
                 colorbalancergb_params=ColorBalanceRGBParams(),
                 filmicrgb_params=FilmicRGBParams()):
    def zineo(x):
        return 0 if x is None else 1

    # Terrible hack - just disabling a stage doesn't let you set params to "",
    # it must be a valid hex string (because it checks version numbers).
    def to_hex(x, default_value):
        return default_value.to_hex_string() if x is None else x.to_hex_string(
        )

    return _FMT_STR.format(
        raw_prepare_params=to_hex(raw_prepare_params, RawPrepareParams()),
        enable_temperature=zineo(temperature_params),
        temperature_params=to_hex(temperature_params, TemperatureParams()),
        enable_highlights=zineo(highlights_params),
        highlights_params=to_hex(highlights_params, HighlightsParams()),
        enable_denoiseprofile=zineo(denoiseprofile_params),
        denoiseprofile_params=to_hex(denoiseprofile_params, DenoiseProfileParams()),
        enable_hazeremoval=zineo(hazeremoval_params),
        hazeremoval_params=to_hex(hazeremoval_params, HazeRemovalParams()),
        enable_exposure=zineo(exposure_params),
        exposure_params=to_hex(exposure_params, ExposureParams()),
        enable_lowpass=zineo(lowpass_params),
        lowpass_params=to_hex(lowpass_params, LowpassParams()),
        enable_censorize=zineo(censorize_params),
        censorize_params=to_hex(censorize_params, CensorizeParams()),
        enable_sharpen=zineo(sharpen_params),
        sharpen_params=to_hex(sharpen_params, SharpenParams()),
        enable_colorbalancergb=zineo(colorbalancergb_params),
        colorbalancergb_params=to_hex(colorbalancergb_params,
                                      ColorBalanceRGBParams()),
        enable_filmicrgb=zineo(filmicrgb_params),
        filmicrgb_params=to_hex(filmicrgb_params, FilmicRGBParams()))


def render(src_dng_path, dst_path, pipe_stage_flags):
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".xmp",
                                     delete=False) as f:
        f.write(get_pipe_xmp(**pipe_stage_flags))
        xmp_path = f.name
    args = [
        _DARKTABLE_CLI, src_dng_path, xmp_path, dst_path, "--core",
        "--disable-opencl", "-d", "perf"
    ]
    print('Running:\n', ' '.join(args), '\n')
    subprocess.run(args)

def get_params_dict(proxy_type, param_name, value, temperature_params, raw_prepare_params, dict=None):

    params_dict = {
        'filmicrgb_params': None,
        'colorbalancergb_params': None,
        'sharpen_params': None,
        'censorize_params': None,
        'lowpass_params': None,
        'exposure_params': None,
        'hazeremoval_params': None,
        'denoiseprofile_params': None,
        'highlights_params': None,
        'temperature_params': temperature_params,
        'raw_prepare_params': raw_prepare_params,
    }

    # Used for input images
    if proxy_type is None:
        return params_dict

    proxy = proxy_type
    param = param_name

    # If a dict is provided, use that instead
    if dict != None:
        params_dict = dict

    # Setting params
    fill_dict = getattr(functions, proxy)
    params_dict = fill_dict(value, params_dict, param=param)

    return params_dict

if __name__ == '__main__':
    proxy_type, param = sys.argv[1].split('_')
    value = float(sys.argv[2])
    dng_path = sys.argv[3]
    output_dir = sys.argv[4]

    # Constants
    image = dng_path.split('\\')[-1]
    print('image: ' + image)

    # Extracting necessary params from the source image
    raw_prepare_params, temperature_params = read_dng_params(dng_path)

    output_path = os.path.join(output_dir, f'{image}_{proxy_type}_{param}')
    output_path = (repr(output_path).replace('\\\\', '/')).strip("'") + f'_{value}.png' # Dealing with Darktable CLI pickiness

    params_dict = get_params_dict(proxy_type, param, value, temperature_params, raw_prepare_params)

    render(dng_path, output_path, params_dict)