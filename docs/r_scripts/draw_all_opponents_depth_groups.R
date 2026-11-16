# =====================================================================
# draw_all_opponents_depth_groups.R
#
# Modified from 611Cdraw_all_opponents.R (OSF code archive) for
# docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 10
# (memory-depth groups, exploratory/outside preregistration).
#
# Unlike the original, step 1 (building the per-group 3-gram files from
# PersonaPD20260611.xlsx) is NOT needed here: scripts/export_depth_groups_for_r.py
# already wrote <CODE>_depthLow.txt / _depthMid.txt / _depthHigh.txt for
# all 5 opponents directly from the ABM's probe data. This script just
# draws all 5 opponents' networks using those files, via
# GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R with
# GROUP_NAMES set to the depth-group names -- no drawing logic changes.
# =====================================================================

# ---- paths ----------------------------------------------------------
# work_dir : holds the <CODE>_depth{Low,Mid,High}.txt files (output of
#            export_depth_groups_for_r.py), the 予約語 files (copy them
#            from docs/r_scripts/vocab/ into work_dir, or point work_dir
#            at that folder directly), and is where the output PNGs land.
work_dir <- "."
# viz_file : full path to the parameterized visualization script. If you
#            copied it into work_dir, the default below works.
viz_file <- file.path(work_dir, "GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R")

setwd(work_dir)

if (!file.exists(viz_file))
  stop(paste0("Viz script not found:\n  ", viz_file,
              "\n-> copy GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R into work_dir, ",
              "or set viz_file to its actual folder."))

GROUP_NAMES <- c("depthLow", "depthMid", "depthHigh")

# ---- draw each opponent (viz script, OPPONENT and GROUP_NAMES swapped) ----
src <- readLines(viz_file, encoding = "UTF-8")
group_names_literal <- paste0(
  "c(", paste0('"', GROUP_NAMES, '"', collapse = ", "), ")"
)

for (opp in c("ALC", "ALD", "GRA", "RAN", "TFT")) {
  cat("\n\n================  drawing network:", opp, " ================\n")
  code <- sub("^OPPONENT *<-.*", sprintf('OPPONENT <- "%s"', opp), src)
  code <- sub("^GROUP_NAMES *<-.*", sprintf("GROUP_NAMES <- %s", group_names_literal), code)
  tryCatch(
    eval(parse(text = code), envir = new.env(parent = globalenv())),
    error = function(e)
      cat("  [skipped]", opp, "->", conditionMessage(e), "\n")
  )
}

cat("\nAll done. One PNG per opponent saved to work_dir.\n")
cat("If an opponent was skipped, check that work_dir has:\n")
cat("  <CODE>_depthLow.txt / _depthMid.txt / _depthHigh.txt (from export_depth_groups_for_r.py)\n")
cat("  GRA予約語64.txt  ALC予約語64.txt  ALD予約語64.txt  RAN予約語64.txt  TFT予約語16.txt (docs/r_scripts/vocab/)\n")
