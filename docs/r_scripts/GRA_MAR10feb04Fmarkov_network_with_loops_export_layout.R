# Markov 3-layer Network Visualization (3 Groups) -- LAYOUT CSV EXPORT
#
# Modified from GRA_MAR10feb04Fmarkov_network_with_loops.R (OSF code
# archive) for evopersonality_abm/EvoPersonality_ABM
# docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 11 (the Jaccard
# threshold x generation HTML viewer's optional fixed node positions,
# exploratory/outside preregistration).
#
# ONLY CHANGE FROM THE ORIGINAL: a few lines right after the unified
# layout is fully computed (normalized, overlaps removed, named) write it
# to <OPPONENT>_layout.csv (columns: node, x, y) -- nothing about the
# layout CALCULATION or the drawing itself changes. Diff against the
# original script to verify: only this header and the "Export layout to
# CSV" block (right after `rownames(layout_unified) <- all_nodes_union`)
# were added.
#
# Supports multiple opponents (TFT, GRA, RAN, ALC, ALD)
# Automatic layout for 64 nodes
# WITH SELF-LOOPS using igraph's built-in rendering

library(igraph)
library(tidyverse)

# ========================================
# Settings
# ========================================
OPPONENT <- "GRA"  # Change to "TFT", "GRA", "RAN", "ALC", "ALD"
JACCARD_THRESHOLD <- 0.30  # Threshold showing clearest group differences
LAYOUT_ALGORITHM <- "fr"  # Layout: "fr" (best for clustered layout), "graphopt", "kk", "drl"
LOOP_ANGLE <- 2*pi/3      # Self-loop angle in radians: 0=top(12 o'clock), pi/2=right(3 o'clock), 
                       # pi=bottom(6 o'clock), 3*pi/2=left(9 o'clock), 4*pi/3=8 o'clock
LOOP_SIZE <- 0.1  # Self-loop size (0.3-0.5 recommended)
LAYOUT_SEED <- 123  # Random seed for reproducible layout (change to get different layouts)
# ========================================

cat("===========================================\n")
cat("Markov Network Analysis (WITH SELF-LOOPS)\n")
cat("Opponent:", OPPONENT, "\n")
cat("Jaccard Threshold:", JACCARD_THRESHOLD, "\n")
cat("Layout Seed:", LAYOUT_SEED, "(for reproducible layouts)\n")
cat("Loop Angle:", LOOP_ANGLE, "rad (", round(LOOP_ANGLE * 180 / pi, 1), "°)\n")
cat("Loop Size:", LOOP_SIZE, "\n")
cat("===========================================\n\n")

# Load reserved words from file
reserved_words_file <- paste0(OPPONENT, "予約語", 
                              ifelse(OPPONENT == "TFT", "16", "64"), ".txt")

cat("Loading reserved words from:", reserved_words_file, "\n")

if (!file.exists(reserved_words_file)) {
  stop(paste("Error: Reserved words file not found:", reserved_words_file))
}

reserved_words <- suppressWarnings(readLines(reserved_words_file, encoding = "UTF-8"))
reserved_words <- trimws(reserved_words)
reserved_words <- reserved_words[nchar(reserved_words) > 0]  # Remove empty lines

cat("Loaded", length(reserved_words), "reserved words\n\n")

# Markov transition extraction function
extract_markov_transitions <- function(file_path) {
  lines <- suppressWarnings(readLines(file_path, encoding = "UTF-8"))
  
  all_nodes <- c()
  transitions <- list()
  word_to_lines <- list()
  
  for (line_num in seq_along(lines)) {
    line <- lines[line_num]
    if (nchar(trimws(line)) == 0) next
    
    nodes <- trimws(strsplit(line, ",")[[1]])
    nodes <- gsub("。$", "", nodes)
    nodes <- unique(nodes[nodes %in% reserved_words])
    
    all_nodes <- c(all_nodes, nodes)
    
    for (node in nodes) {
      if (is.null(word_to_lines[[node]])) {
        word_to_lines[[node]] <- c()
      }
      word_to_lines[[node]] <- c(word_to_lines[[node]], line_num)
    }
    
    if (length(nodes) > 1) {
      for (i in 1:(length(nodes) - 1)) {
        for (j in (i + 1):length(nodes)) {
          node1 <- nodes[i]
          node2 <- nodes[j]
          
          tail1 <- substr(node1, nchar(node1) - 1, nchar(node1))
          head2 <- substr(node2, 1, 2)
          tail2 <- substr(node2, nchar(node2) - 1, nchar(node2))
          head1 <- substr(node1, 1, 2)
          
          if (tail1 == head2) {
            pair_key <- paste(node1, node2, sep = "->")
            if (is.null(transitions[[pair_key]])) {
              transitions[[pair_key]] <- 0
            }
            transitions[[pair_key]] <- transitions[[pair_key]] + 1
          }
          
          if (tail2 == head1) {
            pair_key <- paste(node2, node1, sep = "->")
            if (is.null(transitions[[pair_key]])) {
              transitions[[pair_key]] <- 0
            }
            transitions[[pair_key]] <- transitions[[pair_key]] + 1
          }
        }
      }
    }
  }
  
  # ---------- 自己推移の検出 ----------
  # ノード「ABC」の末尾2文字「BC」が先頭2文字「AB」と一致する場合に自己推移が可能
  # 同じノードが連続して出現する行数をカウント
  for (node in names(word_to_lines)) {
    tail_node <- substr(node, nchar(node) - 1, nchar(node))
    head_node <- substr(node, 1, 2)
    if (tail_node == head_node) {
      # 自己推移が構造的に可能なノード
      # そのノードが出現した行数をカウントする（出現回数＝自己推移の潜在的回数）
      self_key <- paste(node, node, sep = "->")
      n_lines <- length(word_to_lines[[node]])
      if (n_lines > 1) {
        # 2回以上出現した行があれば自己推移として記録
        transitions[[self_key]] <- n_lines - 1
      }
    }
  }
  
  # Calculate Jaccard coefficient
  transitions_with_jaccard <- list()
  
  for (pair_key in names(transitions)) {
    parts <- strsplit(pair_key, "->")[[1]]
    word1 <- parts[1]
    word2 <- parts[2]
    
    lines1 <- word_to_lines[[word1]]
    lines2 <- word_to_lines[[word2]]
    
    common_lines <- length(intersect(lines1, lines2))
    union_lines <- length(union(lines1, lines2))
    
    jaccard <- common_lines / union_lines
    
    transitions_with_jaccard[[pair_key]] <- list(
      word1 = word1,
      word2 = word2,
      count = transitions[[pair_key]],
      jaccard = jaccard
    )
  }
  
  node_freq <- table(all_nodes)
  
  list(
    nodes = all_nodes,
    node_freq = node_freq,
    transitions = transitions_with_jaccard
  )
}

# Convert transitions to data.frame
create_trans_df <- function(trans_list, threshold) {
  df_list <- list()
  for (pair_key in names(trans_list)) {
    item <- trans_list[[pair_key]]
    if (item$jaccard >= threshold) {
      df_list[[length(df_list) + 1]] <- data.frame(
        from = item$word1,
        to = item$word2,
        weight = item$count,
        jaccard = item$jaccard,
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(df_list) > 0) {
    do.call(rbind, df_list)
  } else {
    data.frame(from = character(), to = character(), 
               weight = numeric(), jaccard = numeric(),
               stringsAsFactors = FALSE)
  }
}

# Load data
cat("Loading data files...\n")
cat("Working directory:", getwd(), "\n\n")

file1 <- paste0(OPPONENT, "_A-V-.txt")
file2 <- paste0(OPPONENT, "_mix.txt")
file3 <- paste0(OPPONENT, "_A+V+.txt")

files_to_check <- c(file1, file2, file3)
for (file in files_to_check) {
  if (!file.exists(file)) {
    stop(paste("Error: File not found:", file))
  }
}

pminus_sminus <- extract_markov_transitions(file1)
ps_pm <- extract_markov_transitions(file2)
pplus_splus <- extract_markov_transitions(file3)

cat("Loading complete\n")
cat("A-V- group: ", length(pminus_sminus$node_freq), " nodes, ", 
    length(pminus_sminus$transitions), " transitions\n", sep = "")
cat("mix group: ", length(ps_pm$node_freq), " nodes, ", 
    length(ps_pm$transitions), " transitions\n", sep = "")
cat("A+V+ group: ", length(pplus_splus$node_freq), " nodes, ", 
    length(pplus_splus$transitions), " transitions\n", sep = "")

cat("\nUsing threshold: Jaccard ≥ ", JACCARD_THRESHOLD, "\n", sep = "")

# Extract transition data
pminus_sminus_df <- create_trans_df(pminus_sminus$transitions, JACCARD_THRESHOLD)
ps_pm_df <- create_trans_df(ps_pm$transitions, JACCARD_THRESHOLD)
pplus_splus_df <- create_trans_df(pplus_splus$transitions, JACCARD_THRESHOLD)

# Identify common and unique edges
if (nrow(pminus_sminus_df) > 0 && nrow(ps_pm_df) > 0 && nrow(pplus_splus_df) > 0) {
  edges1 <- paste(pminus_sminus_df$from, pminus_sminus_df$to, sep = "->")
  edges2 <- paste(ps_pm_df$from, ps_pm_df$to, sep = "->")
  edges3 <- paste(pplus_splus_df$from, pplus_splus_df$to, sep = "->")
  
  common_all <- Reduce(intersect, list(edges1, edges2, edges3))
  
  pminus_sminus_df$color <- ifelse(paste(pminus_sminus_df$from, pminus_sminus_df$to, sep = "->") %in% common_all, "red", "blue")
  ps_pm_df$color <- ifelse(paste(ps_pm_df$from, ps_pm_df$to, sep = "->") %in% common_all, "red", "blue")
  pplus_splus_df$color <- ifelse(paste(pplus_splus_df$from, pplus_splus_df$to, sep = "->") %in% common_all, "red", "blue")
  
  cat("\n=== Edge Statistics (after threshold) ===\n")
  cat("A-V- group: ", nrow(pminus_sminus_df), " edges (Common: ", 
      sum(pminus_sminus_df$color == "red"), ", Unique: ", 
      sum(pminus_sminus_df$color == "blue"), ")\n", sep = "")
  cat("mix group: ", nrow(ps_pm_df), " edges (Common: ", 
      sum(ps_pm_df$color == "red"), ", Unique: ", 
      sum(ps_pm_df$color == "blue"), ")\n", sep = "")
  cat("A+V+ group: ", nrow(pplus_splus_df), " edges (Common: ", 
      sum(pplus_splus_df$color == "red"), ", Unique: ", 
      sum(pplus_splus_df$color == "blue"), ")\n", sep = "")
} else {
  cat("\nWarning: No edges found. Try lowering the threshold.\n")
  if (nrow(pminus_sminus_df) == 0) pminus_sminus_df$color <- character()
  if (nrow(ps_pm_df) == 0) ps_pm_df$color <- character()
  if (nrow(pplus_splus_df) == 0) pplus_splus_df$color <- character()
}

# Create graphs
create_graph_from_trans <- function(trans_df, node_freq) {
  if (nrow(trans_df) == 0) {
    g <- make_empty_graph(n = 0, directed = TRUE)
    return(g)
  }
  
  g <- graph_from_data_frame(trans_df[, c("from", "to")], directed = TRUE)
  
  E(g)$weight <- trans_df$weight
  E(g)$jaccard <- trans_df$jaccard
  E(g)$color <- trans_df$color
  
  for (node in V(g)$name) {
    if (node %in% names(node_freq)) {
      V(g)[node]$freq <- as.numeric(node_freq[node])
    } else {
      V(g)[node]$freq <- 1
    }
  }
  
  return(g)
}

graph1 <- create_graph_from_trans(pminus_sminus_df, pminus_sminus$node_freq)
graph2 <- create_graph_from_trans(ps_pm_df, ps_pm$node_freq)
graph3 <- create_graph_from_trans(pplus_splus_df, pplus_splus$node_freq)

# Debug: Check for self-loops in created graphs
cat("\n=== DEBUG: Self-loop detection ===\n")
for (i in 1:3) {
  g <- list(graph1, graph2, graph3)[[i]]
  group_name <- c("A-V-", "mix", "A+V+")[i]
  
  if (ecount(g) > 0) {
    self_loops <- sapply(seq_len(ecount(g)), function(e) {
      ends_e <- ends(g, e)
      ends_e[1] == ends_e[2]
    })
    cat(group_name, "group: Total edges =", ecount(g), 
        ", Self-loops =", sum(self_loops), "\n")
    
    if (sum(self_loops) > 0) {
      cat("  Self-loop nodes:", paste(unique(ends(g, which(self_loops))[,1]), collapse = ", "), "\n")
    }
  }
}
cat("===================================\n\n")

# Layout generation
all_nodes_union <- unique(c(V(graph1)$name, V(graph2)$name, V(graph3)$name))

node_freq_unified <- numeric(length(all_nodes_union))
names(node_freq_unified) <- all_nodes_union

for (node in all_nodes_union) {
  freq_sum <- 0
  if (node %in% names(pminus_sminus$node_freq)) freq_sum <- freq_sum + pminus_sminus$node_freq[node]
  if (node %in% names(ps_pm$node_freq)) freq_sum <- freq_sum + ps_pm$node_freq[node]
  if (node %in% names(pplus_splus$node_freq)) freq_sum <- freq_sum + pplus_splus$node_freq[node]
  node_freq_unified[node] <- freq_sum
}

all_trans_unified <- rbind(
  pminus_sminus_df[, c("from", "to", "weight")],
  ps_pm_df[, c("from", "to", "weight")],
  pplus_splus_df[, c("from", "to", "weight")]
)

graph_unified <- graph_from_data_frame(all_trans_unified[, c("from", "to")], directed = TRUE)

# Function to normalize layout to specific range
normalize_layout <- function(layout, xmin = -1, xmax = 1, ymin = -1, ymax = 1) {
  x <- layout[, 1]
  y <- layout[, 2]
  x_norm <- (x - min(x)) / (max(x) - min(x)) * (xmax - xmin) + xmin
  y_norm <- (y - min(y)) / (max(y) - min(y)) * (ymax - ymin) + ymin
  cbind(x_norm, y_norm)
}

# Function to remove overlaps
remove_overlaps <- function(layout, sizes, min_distance = 1.0, max_iter = 500) {
  n <- nrow(layout)
  for (iter in 1:max_iter) {
    moved <- FALSE
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        dx <- layout[j, 1] - layout[i, 1]
        dy <- layout[j, 2] - layout[i, 2]
        dist <- sqrt(dx^2 + dy^2)
        min_dist <- (sizes[i] + sizes[j]) * min_distance
        
        if (dist < min_dist && dist > 0) {
          move_dist <- (min_dist - dist) / 2
          layout[i, 1] <- layout[i, 1] - (dx / dist) * move_dist
          layout[i, 2] <- layout[i, 2] - (dy / dist) * move_dist
          layout[j, 1] <- layout[j, 1] + (dx / dist) * move_dist
          layout[j, 2] <- layout[j, 2] + (dy / dist) * move_dist
          moved <- TRUE
        }
      }
    }
    if (!moved) break
  }
  return(layout)
}

# Special handling for RAN opponent - hierarchical cascading layout
if (OPPONENT == "RAN") {
  cat("Generating hierarchical cascading layout for RAN opponent...\n")
  
  # Set seed for reproducible layout
  set.seed(LAYOUT_SEED)
  
  components <- components(graph_unified)
  comp_membership <- components$membership
  comp_sizes <- components$csize
  comp_count <- components$no
  
  comp_order <- order(comp_sizes, decreasing = TRUE)
  
  layout_list <- list()
  
  for (comp_idx in comp_order) {
    comp_nodes <- which(comp_membership == comp_idx)
    subgraph <- induced_subgraph(graph_unified, comp_nodes)
    
    if (vcount(subgraph) == 1) {
      sublayout <- matrix(c(0, 0), nrow = 1, ncol = 2)
    } else if (vcount(subgraph) == 2) {
      sublayout <- matrix(c(-0.5, 0, 0.5, 0), nrow = 2, ncol = 2, byrow = TRUE)
    } else {
      sublayout <- layout_with_fr(subgraph, niter = 500)
      sublayout <- normalize_layout(sublayout, xmin = -1, xmax = 1, ymin = -1, ymax = 1)
    }
    
    layout_list[[comp_idx]] <- list(
      nodes = V(subgraph)$name,
      layout = sublayout,
      size = vcount(subgraph)
    )
  }
  
  layout_unified <- matrix(NA, nrow = length(all_nodes_union), ncol = 2)
  rownames(layout_unified) <- all_nodes_union
  
  y_offset <- 0
  y_spacing <- 2.5
  
  for (comp_idx in comp_order) {
    comp_info <- layout_list[[comp_idx]]
    comp_layout <- comp_info$layout
    comp_nodes <- comp_info$nodes
    
    comp_layout[, 2] <- comp_layout[, 2] + y_offset
    
    for (i in seq_along(comp_nodes)) {
      node_name <- comp_nodes[i]
      layout_unified[node_name, ] <- comp_layout[i, ]
    }
    
    y_offset <- y_offset - y_spacing
  }
  
} else {
  cat("Generating clustered layout (cooperation left, defection right)...\n")
  
  # Set seed for reproducible layout
  set.seed(LAYOUT_SEED)
  
  coop_keywords <- c("和", "協")
  defect_keywords <- c("搾", "裏")
  
  coop_nodes <- c()
  defect_nodes <- c()
  neutral_nodes <- c()
  
  for (node in all_nodes_union) {
    if (any(sapply(coop_keywords, function(k) grepl(k, node, fixed = TRUE)))) {
      coop_nodes <- c(coop_nodes, node)
    } else if (any(sapply(defect_keywords, function(k) grepl(k, node, fixed = TRUE)))) {
      defect_nodes <- c(defect_nodes, node)
    } else {
      neutral_nodes <- c(neutral_nodes, node)
    }
  }
  
  layout_init <- matrix(0, nrow = vcount(graph_unified), ncol = 2)
  rownames(layout_init) <- V(graph_unified)$name
  
  for (i in seq_along(V(graph_unified)$name)) {
    node <- V(graph_unified)$name[i]
    if (node %in% coop_nodes) {
      layout_init[i, 1] <- -1 + runif(1, -0.1, 0.1)
      layout_init[i, 2] <- runif(1, -1, 1)
    } else if (node %in% defect_nodes) {
      layout_init[i, 1] <- 1 + runif(1, -0.1, 0.1)
      layout_init[i, 2] <- runif(1, -1, 1)
    } else {
      layout_init[i, 1] <- runif(1, -0.5, 0.5)
      layout_init[i, 2] <- runif(1, -1, 1)
    }
  }
  
  if (LAYOUT_ALGORITHM == "kk") {
    layout_unified <- layout_with_kk(graph_unified, coords = layout_init)
  } else if (LAYOUT_ALGORITHM == "drl") {
    layout_unified <- layout_with_drl(graph_unified)
  } else if (LAYOUT_ALGORITHM == "graphopt") {
    layout_unified <- layout_with_graphopt(graph_unified, niter = 1000)
  } else {
    layout_unified <- layout_with_fr(graph_unified, niter = 500, start = layout_init)
  }
}

# Normalize unified layout
layout_unified <- normalize_layout(layout_unified, xmin = -1, xmax = 1, ymin = -1, ymax = 1)

# Calculate vertex sizes for overlap removal
n_nodes_total <- length(all_nodes_union)
if (n_nodes_total > 30) {
  node_size_factor <- 3.5
} else {
  node_size_factor <- 9.0
}

sizes_unified <- sqrt(node_freq_unified) * node_size_factor * 0.01

# Remove overlaps from unified layout
cat("Removing overlaps from clustered layout...\n")
layout_unified <- remove_overlaps(layout_unified, sizes_unified, min_distance = 3.0)

# Create named layout matrix
rownames(layout_unified) <- all_nodes_union

# ---- Export layout to CSV (added -- see header) ---------------------
layout_csv_path <- paste0(OPPONENT, "_layout.csv")
layout_df <- data.frame(
  node = rownames(layout_unified),
  x = layout_unified[, 1],
  y = layout_unified[, 2],
  stringsAsFactors = FALSE
)
write.csv(layout_df, layout_csv_path, row.names = FALSE, fileEncoding = "UTF-8")
cat("Layout exported to:", layout_csv_path, "(", nrow(layout_df), "nodes )\n")
# -----------------------------------------------------------------------

# Extract layout for each individual graph
extract_layout_for_graph <- function(g, unified_layout) {
  node_names <- V(g)$name
  layout_mat <- matrix(NA, nrow = length(node_names), ncol = 2)
  for (i in seq_along(node_names)) {
    if (node_names[i] %in% rownames(unified_layout)) {
      layout_mat[i, ] <- unified_layout[node_names[i], ]
    }
  }
  return(layout_mat)
}

layout1 <- extract_layout_for_graph(graph1, layout_unified)
layout2 <- extract_layout_for_graph(graph2, layout_unified)
layout3 <- extract_layout_for_graph(graph3, layout_unified)

if (OPPONENT == "RAN") {
  cat("Hierarchical cascading layout complete - components arranged vertically\n\n")
} else {
  cat("Clustered layout complete - cooperation left, defection right\n\n")
}

# Plot function with legend - USING IGRAPH'S BUILT-IN SELF-LOOP RENDERING
plot_network <- function(g, layout, title) {
  if (ecount(g) == 0) {
    plot.new()
    text(0.5, 0.5, "No edges\n(Lower threshold)", cex = 1.5)
    title(title)
    return()
  }
  
  # Adjust node and label size based on number of nodes
  n_nodes <- vcount(g)
  if (n_nodes > 30) {
    node_size_factor <- 2.5
    # RAN opponent needs larger labels due to complexity
    if (exists("OPPONENT") && OPPONENT == "RAN") {
      label_size <- 1.6  # Larger labels for RAN
    } else {
      label_size <- 0.8
    }
  } else {
    node_size_factor <- 6.0
    label_size <- 1.8
  }
  
  V(g)$size <- sqrt(V(g)$freq) * node_size_factor
  E(g)$color_val <- E(g)$color
  E(g)$width <- E(g)$jaccard * 4
  E(g)$arrow_size <- E(g)$jaccard * 1.2
  E(g)$arrow_width <- E(g)$jaccard * 2.0
  
  # Adjust edge curvature based on opponent
  edge_curve <- if (exists("OPPONENT") && OPPONENT == "RAN") 0.1 else 0.25
  
  # Set loop parameters - igraph requires scalar values, not vectors
  # All loops will have the same size and angle
  actual_loop_size <- LOOP_SIZE  # Use the setting directly
  
  # Check if there are any self-loops for debug
  is_loop <- sapply(seq_len(ecount(g)), function(e) {
    ends <- ends(g, e)
    ends[1] == ends[2]
  })
  
  if (any(is_loop)) {
    cat("Debug: Found", sum(is_loop), "self-loops\n")
    cat("Debug: LOOP_SIZE setting =", LOOP_SIZE, "\n")
    cat("Debug: Loop size to be applied =", actual_loop_size, "\n")
  }
  
  plot(g,
       layout = layout,
       vertex.size = V(g)$size,
       vertex.color = "lightblue",
       vertex.frame.color = "black",
       vertex.label = V(g)$name,
       vertex.label.cex = label_size,
       vertex.label.family = "sans",
       vertex.label.color = "black",
       vertex.label.dist = 0,
       edge.color = E(g)$color_val,
       edge.width = E(g)$width,
       edge.arrow.size = E(g)$arrow_size,
       edge.arrow.width = E(g)$arrow_width,
       edge.curved = edge_curve,
       edge.loop.angle = LOOP_ANGLE,      # Scalar value
       edge.loop.size = actual_loop_size,  # Scalar value (not vector!)
       main = "",
       cex.main = 1)
  
  # Panel title
  mtext(title, side = 3, line = 1, cex = 2.2, font = 2)
  
  # Add legend
  legend("bottomright", 
         legend = c("Common transitions", "Group-specific transitions"),
         col = c("red", "blue"),
         lty = 1,
         lwd = 4,
         cex = 1.5,
         bg = "white",
         box.lwd = 1.5)
}

# Save plot
datetime_prefix <- format(Sys.time(), "%Y%m%d_%H%M%S")
filename <- paste0(datetime_prefix, "_markov_network_3groups_", OPPONENT, "_j", 
                   gsub("\\.", "", as.character(JACCARD_THRESHOLD)), "_with_loops.png")

# Adjust image size based on number of nodes and opponent
n_total_nodes <- max(vcount(graph1), vcount(graph2), vcount(graph3))
if (OPPONENT == "RAN") {
  img_width <- 7200
  img_height <- 3600
} else if (n_total_nodes > 30) {
  img_width <- 7200
  img_height <- 2400
} else {
  img_width <- 4500
  img_height <- 1700
}

png(filename, width = img_width, height = img_height, res = 150)
par(mfrow = c(1, 3), mar = c(2, 2, 4, 2), oma = c(3, 0, 0, 0))

# Windows Japanese font setting
if (.Platform$OS.type == "windows") {
  windowsFonts(JP = windowsFont("Meiryo"))
  par(family = "JP")
}

plot_network(graph1, layout1, "A-V- Group")
plot_network(graph2, layout2, "mix Group")
plot_network(graph3, layout3, "A+V+ Group")

# Add overall caption
opponent_full <- switch(OPPONENT,
                        "TFT" = "TFT (Tit-for-Tat)",
                        "GRA" = "GRA (Gradual)",
                        "RAN" = "RAN (Random)",
                        "ALC" = "ALC (All-C)",
                        "ALD" = "ALD (All-D)",
                        OPPONENT)

mtext(paste0("Opponent: ", opponent_full, "  |  Jaccard Threshold ≥ ", JACCARD_THRESHOLD),
      side = 1, line = 1, outer = TRUE, cex = 2.2, font = 2)

dev.off()
cat("\nFigure saved:", filename, "\n")
cat("Image size:", img_width, "x", img_height, "pixels\n")
cat("Number of nodes: Group1 =", vcount(graph1), ", Group2 =", vcount(graph2), 
    ", Group3 =", vcount(graph3), "\n\n")
cat("=== SELF-LOOP INFORMATION ===\n")
cat("Self-loop angle:", LOOP_ANGLE, "radians (", round(LOOP_ANGLE * 180 / pi, 1), "degrees)\n")
cat("Self-loop size:", LOOP_SIZE, "\n")
cat("Self-loops detected for nodes where head = tail (e.g., 搾搾搾, 和和和)\n")
cat("To change loop position, modify LOOP_ANGLE at the top of the script\n")
cat("  0 = 12 o'clock (top)\n")
cat("  pi/2 = 3 o'clock (right)\n")
cat("  pi = 6 o'clock (bottom)\n")
cat("  3*pi/2 = 9 o'clock (left)\n")
cat("  4*pi/3 = 8 o'clock\n")
cat("  5*pi/3 = 4 o'clock\n\n")
cat("To change opponent, modify OPPONENT at the top of the script.\n")
cat("Available opponents: TFT, GRA, RAN, ALC, ALD\n")
