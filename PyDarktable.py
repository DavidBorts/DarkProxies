import io
import os
import math
import numpy as np
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
      darktable:modversion="5"
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
      <rdf:li
      darktable:num="16"
      darktable:operation="graduatednd"
      darktable:enabled="{enable_graduateddensity}"
      darktable:modversion="1"
      darktable:params="{graduateddensity_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz10eJxjYGBgYAFiCQYYOOHEgAZY0QVwggZ7CB6pfOygYtaVAyCMi08IAAB/xiOk"/>
      <rdf:li
      darktable:num="17"
      darktable:operation="bloom"
      darktable:enabled="{enable_bloom}"
      darktable:modversion="1"
      darktable:params="{bloom_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz13eJxjYGBgYAJiCQYYOOHEgAZY0QVwggZ7CB6pfNoAAFJgGQo="/>
      <rdf:li
      darktable:num="18"
      darktable:operation="colorize"
      darktable:enabled="{enable_colorize}"
      darktable:modversion="2"
      darktable:params="{colorize_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz13eJxjYGBgYAJiCQYYOOHEgAZY0QVwggZ7CB6pfNoAAFJgGQo="/>
      <rdf:li
      darktable:num="19"
      darktable:operation="soften"
      darktable:enabled="{enable_soften}"
      darktable:modversion="1"
      darktable:params="{soften_params}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="11"
      darktable:blendop_params="gz10eJxjYGBgYAFiCQYYOOHEgAZY0QVwggZ7CB6pfOygYtaVAyCMi08IAAB/xiOk"/>
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

    # v4 params
    vibrance: float = 0.0  # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0 $DESCRIPTION: "global vibrance"
    grey_fulcrum: float = 0.1845  # $MIN:  0.0 $MAX: 1.0 $DEFAULT: 0.1845 $DESCRIPTION: "contrast gray fulcrum"
    contrast: float = 0.0  # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0. $DESCRIPTION: "contrast"

    # v5 params (current)
    saturation_formula: int = 1 # $DEFAULT: 1 $DESCRIPTION: "saturation formula"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

# dt_iop_exposure_params_t
@dataclass
class ExposureParams:
    # In C, they are enums:
    # MANUAL = 0     "manual"
    # DEFLICKER = 1  "automatic"
    mode: int = 0

    black: float = 0.0 # $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "black level correction"
    exposure: float = 0.0 #  $MIN: -18.0 $MAX: 18.0 $DEFAULT: 0.0
    deflicker_percentile: float = 50.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "percentile"
    deflicker_target_level: float = -4.0 # $MIN: -18.0 $MAX: 18.0 $DEFAULT: -4.0 $DESCRIPTION: "target level"

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
    radius: float = 8.0 # $MIN: 0.0 $MAX: 99.0 $DEFAULT: 2.0
    amount: float = 0.5 # $MIN: 0.0 $MAX: 2.0 $DEFAULT: 0.5
    threshold: float = 0.5 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 0.5

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
    red: float = 1.420689582824707 # $MIN: 0.0 $MAX: 8.0
    green: float = 1.0 # $MIN: 0.0 $MAX: 8.0
    blue: float = 2.0 # $MIN: 0.0 $MAX: 8.0
    g2: float = math.nan

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])
    
@dataclass
class ColorizeParams:
    hue: float = 0.0 # $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
    saturation: float = 0.5 # $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5
    source_lightness_mix: float = 50.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "source mix"
    lightness: float = 50.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class GraduatedDensityParams:
    density: float = 1.0 # $MIN: -8.0 $MAX: 8.0 $DEFAULT: 1.0 $DESCRIPTION: "density" The density of filter 0-8 EV
    hardness: float = 0.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 0.0 $DESCRIPTION: "hardness" 0% = soft and 100% = hard
    rotation: float = 0.0 # $MIN: -180.0 $MAX: 180.0 $DEFAULT: 0.0 $DESCRIPTION: "rotation" 2*PI -180 - +180
    offset: float = 50.0 # $DEFAULT: 50.0 $DESCRIPTION: "offset" centered, can be offsetted...
    hue: float = 0.0 # $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "hue"
    saturation: float = 0.0 # $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "saturation"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

@dataclass
class SoftenParams:
    size: float = 50.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0
    saturation: float = 100.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 100.0
    brightness: float = 0.33 # $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.33
    amount: float = 50.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "mix"

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])
    
@dataclass
class BloomParams:
    size: float = 20.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 20.0
    threshold: float = 90.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 90.0
    strength: float = 25.0 # $MIN: 0.0 $MAX: 100.0 $DEFAULT: 25.0

    def to_hex_string(self):
        return to_hex_string([getattr(self, fd.name) for fd in fields(self)])

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

def fill(params_dict, proxy_type, params, values):
    ops = {
        "colorbalancergb": ColorBalanceRGBParams,
        "highlights": HighlightsParams,
        "sharpen": SharpenParams,
        "exposure": ExposureParams,
        "graduateddensity": GraduatedDensityParams,
        "hazeremoval": HazeRemovalParams,
        "denoiseprofile": DenoiseProfileParams,
        "lowpass": LowpassParams,
        "censorize": CensorizeParams,
        "filmicrgb": FilmicRGBParams,
        "bloom": BloomParams,
        "colorize": ColorizeParams,
        "soften": SoftenParams,
        "temperature": TemperatureParams
    }
    params_class = ops[proxy_type]()

    #TODO: replace enumerate w/ zip)() ??
    for i, param in enumerate(params):
        setattr(params_class, param, float(values[i]))

    params_dict[f"{proxy_type}_params"] = params_class
    return params_dict
        
# Pipeline order is as follows, * means it can be skipped and therefore has a bool param.
# 0 rawprepare
# 1 temperature       *
# 2 highlights        *
# 3 demosaic
# 4 denoise(profiled) *
# 5 hazeremoval       *
# 6 flip              *
# 7 exposure          *
# 8 graduated density *
# 9 colorin
# 10 censorize        *
# 11 lowpass          *
# 12 sharpen          *
# 13 colorbalancergb  *
# 14 filmicrgb        *
# 15 bloom            *
# 16 colorize         *
# 17 soften           *
# 1X colorout
def get_pipe_xmp(raw_prepare_params=RawPrepareParams(),
                 temperature_params=TemperatureParams(),
                 highlights_params=HighlightsParams(),
                 denoiseprofile_params=DenoiseProfileParams(),
                 hazeremoval_params=HazeRemovalParams(),
                 exposure_params=ExposureParams(),
                 graduateddensity_params=GraduatedDensityParams(),
                 censorize_params=CensorizeParams(),
                 lowpass_params=LowpassParams(),
                 sharpen_params=SharpenParams(),
                 colorbalancergb_params=ColorBalanceRGBParams(),
                 filmicrgb_params=FilmicRGBParams(),
                 bloom_params=BloomParams(),
                 colorize_params=ColorizeParams(),
                 soften_params=SoftenParams()):
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
        enable_graduateddensity=zineo(graduateddensity_params),
        graduateddensity_params=to_hex(graduateddensity_params, GraduatedDensityParams()),
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
        filmicrgb_params=to_hex(filmicrgb_params, FilmicRGBParams()),
        enable_bloom=zineo(bloom_params),
        bloom_params=to_hex(bloom_params, BloomParams()),
        enable_colorize=zineo(colorize_params),
        colorize_params=to_hex(colorize_params, ColorizeParams()),
        enable_soften=zineo(soften_params),
        soften_params=to_hex(soften_params, SoftenParams()))

def extract_pfm(log, module):
    for line in log.stdout.split('\n'):
            tmp_dir = line.split('\'')[-1]
            pfm_files = os.listdir(tmp_dir)
            module_tapouts = [pfm_file for pfm_file in pfm_files 
                              if pfm_file.contains(str(module)) 
                              and not pfm_file.contains("diff")]
            return module_tapouts

def pfm_to_tif(pfm_path, dest_path):
    args = [c.MAGICK_COMMAND, pfm_path, dest_path]
    print('Running:\n', ' '.join(args), '\n')
    subprocess.run(args)

def render(src_dng_path, dst_path, pipe_stage_flags, tapout, module=None):
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".xmp",
                                     delete=False) as f:
        f.write(get_pipe_xmp(**pipe_stage_flags))
        xmp_path = f.name
    args = [
        _DARKTABLE_CLI, src_dng_path, xmp_path, dst_path, "--hq", "true", "--core",
        "--disable-opencl", "-d", "perf"
    ]
    if tapout:
        args += ["--dump-pipe", str(module)]
    print('Running:\n', ' '.join(args), '\n')
    result = subprocess.run(args, capture_output=True, text=True)
    if tapout:
        tapout_in, tapout_out = extract_pfm(result, module)
        return (tapout_in, tapout_out)
        

def get_params_dict(proxy_type, param_names, values, temperature_params, raw_prepare_params, dict=None):

    params_dict = {
        'soften_params': None,
        'colorize_params': None,
        'bloom_params': None,
        'filmicrgb_params': None,
        'colorbalancergb_params': None,
        'sharpen_params': None,
        'censorize_params': None,
        'lowpass_params': None,
        'exposure_params': None,
        'graduateddensity_params': None,
        'hazeremoval_params': None,
        'denoiseprofile_params': None,
        'highlights_params': None,
        'temperature_params': temperature_params,
        'raw_prepare_params': raw_prepare_params,
    }

    # Used for input images
    if proxy_type is None:
        return params_dict

    #TODO: is this necessary?
    proxy = proxy_type
    params = param_names

    # If a dict is provided, use that instead of creating
    # a new one from scratch
    if dict != None:
        params_dict = dict

    # Setting params
    params_dict = fill(params_dict, proxy, params, values)
    return params_dict