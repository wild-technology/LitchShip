{
  "generator": "LitchShip Progressive Pipeline",
  "version": "1.0",
  "component": {
    "medianError": $(componentMedianError),
    "meanError": $(componentMeanError),
    "avgTrackLength": $(componentAverageTrackLength),
    "totalProjections": $(componentTotalProjection),
    "cameraCount": $(cameraCount)
  },
  "cameras": [
$IterateCameras($If( $(index) > 0, COMMA_PLACEHOLDER)
    {
      "index": $(index),
      "image": "$(imageName).$(imageExt)",
      "width": $(width),
      "height": $(height),
$CameraErrors( cameraIndex,
      "numPoints": $(numPoints),
      "imageCoverage": $(imageCoverage),
      "reprojError": $ReprojectionError({
        "median": $(median),
        "mean": $(mean),
        "max": $(max),
        "stdev": $(stdev),
        "mode": $(mode)
      })
)
    })
  ]
}
<!--
  RealityScan Report Template — Quality Report for Progressive Training
  =====================================================================

  This template extracts per-image quality metrics from a RealityScan
  project and outputs them as JSON for use by the LitchShip progressive
  training pipeline (analyze_colmap.py).

  INSTALLATION (automated):

    python install_template.py
    python install_template.py --rs-path "D:\Epic Games\RealityScan"

  INSTALLATION (manual):

  1. Copy this file to RealityScan's Reports/ directory:
     C:\Program Files\Epic Games\RealityScan\Reports\quality_report.json.tpl

  2. Add the following entry to Report.xml in the RealityScan install directory
     (inside the root <formats> element):

     <format id="{8F3A1B2C-4D5E-6F78-9A0B-C1D2E3F4A5B6}" mask="*.json" desc="Quality Report (JSON) for LitchShip" writer="RealityScan.Export.ReportWriter">
       <hint>Per-image quality metrics (tie points, coverage, reprojection error) for progressive Gaussian splatting training</hint>
       <body>$Include("Reports\quality_report.json.tpl")</body>
     </format>

  EXPORT:

    GUI:  ALIGNMENT tab -> Export -> Report -> select "Quality Report (JSON) for LitchShip"
    CLI:  RealityScan.exe -load project.rsproj -exportReport "C:\output\quality_report.json" "{8F3A1B2C-4D5E-6F78-9A0B-C1D2E3F4A5B6}"

  POST-PROCESSING:

    The template uses COMMA_PLACEHOLDER for JSON array separators because
    RealityScan's $If function may conflict with literal commas as arguments.
    Run the post-processor to produce valid JSON:

      python fix_report_json.py C:\output\quality_report.json

    Or simply pass the raw file to analyze_colmap.py — it cleans up automatically.

  TEMPLATE FUNCTION REFERENCE:

    $IterateCameras(body)
      Iterates all registered cameras in the selected component.
      Exposes: $(index), $(imageName), $(imageExt), $(width), $(height)

    $CameraErrors(cameraIndex, body)
      Per-camera tie-point statistics for the camera at cameraIndex.
      Exposes: $(numPoints), $(imageCoverage)

    $ReprojectionError(body)
      Reprojection error distribution (nested inside $CameraErrors).
      Exposes: $(median), $(mean), $(min), $(max), $(stdev), $(mode)

    $If(condition, trueText)
      Conditional output. Used here to prepend commas between JSON array elements.

    $(componentMedianError), $(componentMeanError), etc.
      Component-level aggregate statistics (Basic/Component scope).

  See: https://rshelp.capturingreality.com/en-US/appbasics/reports_fav_cameras.htm
-->
