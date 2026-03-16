{
  "generator": "LitchShip RealityScan Progressive Pipeline",
  "version": "1.0",
  "component": {
    "medianError": $componentMedianError,
    "meanError": $componentMeanError,
    "avgTrackLength": $componentAverageTrackLength,
    "totalProjections": $componentTotalProjection
  },
  "cameras": [
$IterateCameras(    {
      "index": $(index),
      "image": "$(imageName).$(imageExt)",
      "width": $(width),
      "height": $(height),
      "numPoints": $CameraErrors($(numPoints)),
      "imageCoverage": $CameraErrors($(imageCoverage)),
      "reprojError": $CameraErrors($ReprojectionError({
        "median": $(median),
        "mean": $(mean),
        "max": $(max),
        "stdev": $(stdev)
      }))
    })
  ]
}

<!--
  RealityScan Report Template — Quality Report for Progressive Training
  =====================================================================

  INSTALLATION:

  1. Copy this file to your RealityScan installation's Reports/ directory:
     C:\Program Files\Epic Games\RealityScan\Reports\quality_report.json.tpl

  2. Add the following entry to Report.xml in the same directory:

     <report>
       <guid>{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}</guid>
       <filemask>*.json</filemask>
       <description>Quality Report (JSON) for LitchShip Progressive Training</description>
       <body>quality_report.json.tpl</body>
     </report>

  3. Export via GUI:  ALIGNMENT tab → Export → Report → select "Quality Report (JSON)"
     Export via CLI:  -exportReport "C:\output\quality_report.json" "{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"

  NOTES:
  - The $IterateCameras function iterates over all registered cameras in the component
  - $CameraErrors exposes per-image tie-point statistics
  - $ReprojectionError provides pixel-level error distribution within $CameraErrors
  - Comma handling: RealityScan's template engine handles array separators automatically
    when using $IterateCameras. If your export has trailing commas, post-process with:
      python3 -c "import json,sys; json.dump(json.load(open(sys.argv[1])),open(sys.argv[1],'w'),indent=2)" quality_report.json
-->
