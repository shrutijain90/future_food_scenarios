library(tidyverse)
library(ggrepel)
library(janitor)
library(ragg)
library(scales)
library(ggforce)
library(patchwork)
library(tidytext)
library(grid) 
library(stringr)
library(ggh4x)
library(dplyr)
library(cowplot)

setwd('../../OPSIS/Data/') 

###-----------------------------------------------------------###
### GLOBAL AGGREGATES ###
###-----------------------------------------------------------###

df <- readr::read_csv("fig_data/global_agg.csv", show_col_types = FALSE) %>%
  mutate(
    food_group = factor(food_group, levels = c(
      "Grains",
      "Roots and tubers",
      "Fruits and vegetables",
      "Soybeans",
      "Legumes, nuts and seeds",
      "Oilcrops and sugar crops"
    )),
    diet_scn = factor(diet_scn, levels = c("BMK","FLX","PSC","VEG","VGN")),
    lib_scn  = factor(lib_scn, levels = c("high","low"),
                      labels = c("Low-constraint","High-constraint")),
    metric_lab = recode(metric,
                        demand = "Production (million tonnes)",
                        fraction_trade = "Fraction traded (%)"),
    plot_value = if_else(metric == "fraction_trade", value * 100, value)
  )


diet_cols <- c(
  BMK="#CC3311", FLX="#EE7733",
  PSC="#0077BB", VEG="#33BBEE", VGN="#117733"
)

base_theme <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.title = element_blank()
  )

tight <- theme(plot.margin = margin(4, 6, 4, 6))

# Row strip (centered horizontally)
row_strip <- function(txt) {
  ggplot() + theme_void() +
    annotate("text", x = 0.5, y = 0.5, label = txt,
             hjust = 0.5, vjust = 0.25, size = 3.5) +
    theme(plot.margin = margin(0, 0, 3, 0))
}

# Group title
group_title <- function(txt) {
  ggplot() + theme_void() +
    labs(title = txt) +
    theme(plot.title = element_text(size = 13, face = "bold", hjust = -0.05),
          plot.margin = margin(4, 0, 0, 0))
}

# Small helper to build one panel
mk_panel <- function(dat,
                     ylab = NULL, xlab = NULL,
                     ylim = NULL,
                     show_xticks = TRUE, show_yticks = TRUE,
                     show_color_legend = FALSE, show_shape_legend = FALSE,
                     y_ticks = 4) {                        # <- NEW: desired tick count
  
  y_breaks <- scales::pretty_breaks(n = y_ticks)
  
  p <- ggplot(dat,
              aes(x = year, y = plot_value,
                  color = diet_scn, shape = factor(RCP),
                  group = interaction(diet_scn, RCP))) +
    geom_line(linewidth = 0.5) +
    geom_point(size = 2.5) +
    scale_color_manual(values = diet_cols) +
    scale_shape_manual(values = c(`2.6` = 16, `7` = 17)) +
    labs(x = xlab, y = ylab) +
    base_theme +
    theme(
      panel.grid.minor = element_blank(),
      axis.text.x  = if (show_xticks) element_text() else element_blank(),
      axis.ticks.x = if (show_xticks) element_line() else element_blank(),
      axis.text.y  = if (show_yticks) element_text() else element_blank(),
      axis.ticks.y = if (show_yticks) element_line() else element_blank()
    )
  
  if (is.null(ylim)) {
    p <- p + scale_y_continuous(breaks = y_breaks)
  } else {
    p <- p + scale_y_continuous(limits = ylim,
                                breaks = y_breaks,
                                expand = expansion(mult = c(0.02, 0.05)))
  }
  
  if (!show_color_legend) p <- p + guides(color = "none")
  if (!show_shape_legend) p <- p + guides(shape = "none")
  
  p
}


#  build one 2×2 block; show_legend only for the first group 
build_group_block <- function(g, show_legend = FALSE) {
  g_dat <- df %>% filter(food_group == g)
  
  ylim_dem  <- range(g_dat$plot_value[g_dat$metric == "demand"], na.rm = TRUE)
  ylim_frac <- range(g_dat$plot_value[g_dat$metric == "fraction_trade"], na.rm = TRUE)
  
  d_high <- g_dat %>% filter(lib_scn == "High-constraint", metric == "demand")
  f_high <- g_dat %>% filter(lib_scn == "High-constraint", metric == "fraction_trade")
  d_low  <- g_dat %>% filter(lib_scn == "Low-constraint",  metric == "demand")
  f_low  <- g_dat %>% filter(lib_scn == "Low-constraint",  metric == "fraction_trade")
  
  # Only the top-right carries color legend and bottom-right carries shape legend
  keep_color <- show_legend
  keep_shape <- show_legend
  
  p11 <- mk_panel(d_high,
                  ylab = "Production (million tonnes)", xlab = NULL,
                  ylim = ylim_dem, show_xticks = FALSE, show_yticks = TRUE,
                  show_color_legend = FALSE, show_shape_legend = FALSE)
  
  p12 <- mk_panel(f_high,
                  ylab = "Fraction traded (%)", xlab = NULL,
                  ylim = ylim_frac, show_xticks = FALSE, show_yticks = TRUE,
                  show_color_legend = keep_color, show_shape_legend = FALSE)
  
  p21 <- mk_panel(d_low,
                  ylab = "Production (million tonnes)", xlab = "Year",
                  ylim = ylim_dem, show_xticks = TRUE, show_yticks = TRUE,
                  show_color_legend = FALSE, show_shape_legend = FALSE)
  
  p22 <- mk_panel(f_low,
                  ylab = "Fraction traded (%)", xlab = "Year",
                  ylim = ylim_frac, show_xticks = TRUE, show_yticks = TRUE,
                  show_color_legend = FALSE, show_shape_legend = keep_shape)
  
  gt   <- group_title(as.character(g))
  r_hi <- row_strip("Trade Constraints: High")
  r_lo <- row_strip("Trade Constraints: Low")
  
  gt / r_hi / (p11 | p12) / r_lo / (p21 | p22) +
    plot_layout(heights = c(0.12, 0.08, 1, 0.08, 1), widths = c(1, 1))
}

# Build all groups and arrange 2 columns × 3 rows 
groups <- levels(df$food_group)

blocks <- purrr::map2(groups, seq_along(groups),
                      ~ build_group_block(.x, show_legend = .y == 1))

# Helper: add a center gutter inside a row
add_center_gutter <- function(left_block, right_block, gap = 0.06) {
  (left_block | plot_spacer() | right_block) +
    plot_layout(widths = c(1, gap, 1))
}

# Build rows (1..3) with a center gutter
row1 <- add_center_gutter(blocks[[1]], blocks[[2]], gap = 0.06)
row2 <- add_center_gutter(blocks[[3]], blocks[[4]], gap = 0.06)
row3 <- add_center_gutter(blocks[[5]], blocks[[6]], gap = 0.06)

# Vertical gap between rows (relative height of spacer rows)
row_gap <- 0.02  

final <- row1 / plot_spacer() / row2 / plot_spacer() / row3 +
  plot_layout(heights = c(1, row_gap, 1, row_gap, 1), guides = "collect") &
  theme(
    legend.position   = "right",
    legend.title      = element_text(size = 13, face = "bold"),
    legend.text       = element_text(size = 13),
    legend.key.width  = unit(30, "pt"),
    legend.key.height = unit(25, "pt"),      
    legend.spacing.y  = unit(30, "pt")
  ) &
  guides(
    color = guide_legend(
      title       = "Diet",
      keyheight   = unit(20, "pt"),         
      byrow       = TRUE,
      override.aes = list(size = 3.5)
    ),
    shape = guide_legend(
      title       = "RCP",
      keyheight   = unit(20, "pt"),          
      byrow       = TRUE,
      override.aes = list(size = 3.5)
    )
  )

print(final)

ggsave("fig_data/global_agg.png", final, width = 14, height = 18, dpi = 300)







###-----------------------------------------------------------###
### PERCENTAGE CHANGE ###
###-----------------------------------------------------------###

# Load data
df <- read_csv("fig_data/perc_change.csv")

# Filter to 2030 and 2050, average across climate × trade scenarios
df_summary <- df %>%
  filter(year %in% c(2030, 2050)) %>%
  group_by(year, diet_scn, food_group) %>%
  summarise(
    supply_growth = mean(supply_growth) * 100,  # Convert to %
    trade_growth = mean(trade_growth) * 100,
    .groups = "drop"
  ) %>%
  # Order diets and commodities
  mutate(
    diet_scn = factor(diet_scn, levels = c("VGN", "VEG", "PSC", "FLX", "BMK")),
    food_group = factor(food_group, levels = c(
      "Grains", "Roots and tubers", 
      "Fruits and vegetables", "Soybeans",
      "Legumes, nuts and seeds",  "Oilcrops and sugar crops"
    ))
  )

# Reshape wide for dumbbell
df_wide <- df_summary %>%
  pivot_wider(
    names_from = year,
    values_from = c(supply_growth, trade_growth),
    names_sep = "_"
  )

# long tables for points to map year -> shape
df_long_supply <- df_summary %>%
  select(year, diet_scn, food_group, supply_growth) %>%
  rename(growth = supply_growth) %>%
  mutate(year = as.character(year))

df_long_trade <- df_summary %>%
  select(year, diet_scn, food_group, trade_growth) %>%
  rename(growth = trade_growth) %>%
  mutate(year = as.character(year))

# color palette 
diet_colors <- c(BMK="#CC3311", FLX="#EE7733", PSC="#0077BB", VEG="#33BBEE", VGN="#117733")

# common shape scale and legend override to make year symbols larger
shape_scale <- scale_shape_manual(
  values = c("2030" = 21, "2050" = 19),
  name = 'Year'
)

shape_guide <- guides(
  shape = guide_legend(
    override.aes = list(size = 2)  # 
  )
)

# --- Supply panel ---
p_supply <- ggplot() +
  # segment lines (from wide) connecting 2030 -> 2050
  geom_segment(
    data = df_wide,
    aes(x = supply_growth_2030, xend = supply_growth_2050,
        y = diet_scn, yend = diet_scn, color = diet_scn),
    linewidth = 0.8, alpha = 0.3
  ) +
  # points (from long), map shape to year
  geom_point(
    data = df_long_supply,
    aes(x = growth, y = diet_scn, color = diet_scn, shape = year),
    size = 2, stroke = 1, fill = "white"
  ) +
  facet_wrap(~ food_group, scales = "free_x", ncol = 1) +
  scale_color_manual(values = diet_colors, breaks = c("BMK", "FLX", "PSC", "VEG", "VGN"), name = "Diet") +
  shape_scale +
  shape_guide +
  labs(x = NULL, y = NULL, title = "Production growth from 2020 (%)") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "none",   
    plot.margin = margin(5, 15, 5, 5),
    panel.spacing.y = unit(2, "lines")
  )

# --- Trade panel ---
p_trade <- ggplot() +
  geom_segment(
    data = df_wide,
    aes(x = trade_growth_2030, xend = trade_growth_2050,
        y = diet_scn, yend = diet_scn, color = diet_scn),
    linewidth = 0.8, alpha = 0.3
  ) +
  geom_point(
    data = df_long_trade,
    aes(x = growth, y = diet_scn, color = diet_scn, shape = year),
    size = 2, stroke = 1, fill = "white"
  ) +
  facet_wrap(~ food_group, scales = "free_x", ncol = 1) +
  scale_color_manual(values = diet_colors, breaks = c("BMK", "FLX", "PSC", "VEG", "VGN"), name = "Diet") +
  shape_scale +
  shape_guide +
  labs(x = NULL, y = NULL, title = "Trade growth from 2020 (%)") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 9),
    axis.text.y = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(5, 5, 5, 15),
    panel.spacing.y = unit(2, "lines")
  )

# --- Combine and collect legends ---
p_final <- (p_supply + plot_spacer() + p_trade) +
  plot_layout(widths = c(1, 0.08, 1), guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = 'vertical',
    legend.spacing.y = unit(-10, "pt"),
    plot.caption = element_text(hjust = 0.5, size = 10, margin = margin(t = 6))
  )

print(p_final)
ggsave("fig_data/production_trade_growth.png", p_final, width = 8, height = 10, dpi = 300)






###-----------------------------------------------------------###
### 2020 BASELINE SUPPLY, IMPORT AND EXPORT VOLUMES AND SHARES ###
###-----------------------------------------------------------###

df <- read_csv("fig_data/country_fractions.csv") %>%
  mutate(diet_scn = factor(diet_scn, levels = c("BMK","FLX","PSC","VEG","VGN")))

region_order <- c("CNE", "EAP", "EUR", "LAC", "MEN", "NAM", "SAS", "SSA")
fg_order     <- c("Grains", "Roots and tubers",
                  "Fruits and vegetables", "Soybeans",
                  "Legumes, nuts and seeds", "Oilcrops and sugar crops")

# 2020 baseline (average across the repeated scenarios) 
bl <- df %>%
  filter(year == 2020) %>%
  group_by(region_group_IMPACT, food_group) %>%
  summarise(
    vol_sup = mean(supply) / 1e3,   # kt to Mt
    vol_imp = mean(`import`) / 1e3,
    vol_exp = mean(export)  / 1e3,
    sh_sup  = mean(perc_sup),
    sh_imp  = mean(perc_imp),
    sh_exp  = mean(perc_exp),
    .groups = "drop"
  ) %>%
  mutate(
    region_group_IMPACT = factor(region_group_IMPACT, levels = rev(region_order)),
    food_group = factor(food_group, levels = fg_order)
  )

# Shared limits per row 
vol_max <- max(bl$vol_sup, bl$vol_imp, bl$vol_exp)
sh_max  <- max(bl$sh_sup, bl$sh_imp, bl$sh_exp)

# Heatmap builder
make_heatmap <- function(data, value_col, title,
                         fmt = "%.0f", palette = "seq",
                         lim_max = NULL, show_y = TRUE) {
  
  vals <- data[[value_col]]
  lims <- c(0, ifelse(is.null(lim_max), max(vals), lim_max))
  
  if (palette == "seq") {
    fill_scale <- scale_fill_distiller(
      palette = "YlOrRd", direction = 1,
      limits = lims, name = NULL
    )
  } else {
    fill_scale <- scale_fill_distiller(
      palette = "Blues", direction = 1,
      limits = lims, name = NULL
    )
  }
  
  p <- data %>%
    ggplot(aes(y = region_group_IMPACT, x = food_group,
               fill = .data[[value_col]])) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(
      aes(
        label = sprintf(fmt, .data[[value_col]]),
        colour = ifelse(.data[[value_col]] > quantile(vals, 0.75),
                        "high", "low")
      ),
      size = 3, show.legend = FALSE
    ) +
    scale_colour_manual(values = c("high" = "white", "low" = "grey25")) +
    fill_scale +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x       = element_text(size = 9, angle = 45, hjust = 1),
      axis.text.y       = element_text(size = 9),
      panel.grid        = element_blank(),
      legend.position   = "bottom",
      legend.key.height = unit(0.25, "cm"),
      legend.key.width  = unit(1.8, "cm"),
      legend.margin     = margin(t = -5, b = 0),
      plot.title        = element_text(face = "bold", size = 11)
    )
  
  if (!show_y) p <- p + theme(axis.text.y = element_blank())
  p
}

# Top row: volumes
v_sup <- make_heatmap(bl, "vol_sup", "Supply",  fmt="%.0f", palette="seq",  lim_max=vol_max, show_y=TRUE)
v_imp <- make_heatmap(bl, "vol_imp", "Imports", fmt="%.0f", palette="seq",  lim_max=vol_max, show_y=FALSE)
v_exp <- make_heatmap(bl, "vol_exp", "Exports", fmt="%.0f", palette="seq",  lim_max=vol_max, show_y=FALSE)

# Bottom row: shares
s_sup <- make_heatmap(bl, "sh_sup", "Supply",  fmt="%.1f", palette="blues", lim_max=sh_max, show_y=TRUE)
s_imp <- make_heatmap(bl, "sh_imp", "Imports", fmt="%.1f", palette="blues", lim_max=sh_max, show_y=FALSE)
s_exp <- make_heatmap(bl, "sh_exp", "Exports", fmt="%.1f", palette="blues", lim_max=sh_max, show_y=FALSE)

# Row labels 
vol_label <- wrap_elements(
  textGrob("Volume (Mt)", gp = gpar(fontsize = 12, fontface = "bold"))
)
sh_label <- wrap_elements(
  textGrob("Share of global total (%)", gp = gpar(fontsize = 12, fontface = "bold"))
)

# Assemble 2 × 3 
top_row    <- (v_sup | v_imp | v_exp) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
bottom_row <- (s_sup | s_imp | s_exp) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

fig <- (vol_label / top_row / sh_label / bottom_row) +
  plot_layout(heights = c(0.08, 1, 0.2, 1)) &
  theme(plot.margin = margin(4, 6, 4, 6))

print(fig)
ggsave("fig_data/baseline_2020_shares.png", fig, width = 10, height = 9, dpi = 300)







###-----------------------------------------------------------###
### HEATMAPS OF CHANING SUPPLY, IMPORT AND EXPORT SHARES ###
###-----------------------------------------------------------###

df <- read_csv("fig_data/country_fractions.csv") %>%
  mutate(
    diet_scn = factor(diet_scn, levels = c("BMK", "FLX", "PSC", "VEG", "VGN"))
  )

region_order <- c("CNE", "EAP", "EUR", "LAC", "MEN", "NAM", "SAS", "SSA")
fg_order     <- c("Grains", "Roots and tubers", 
                  "Fruits and vegetables", "Soybeans",
                  "Legumes, nuts and seeds",  "Oilcrops and sugar crops")

# Midpoints across RCP × lib per year × diet × region × food
df_mid <- df %>%
  group_by(region_group_IMPACT, year, diet_scn, food_group) %>%
  summarise(sup = mean(perc_sup),
            imp = mean(perc_imp),
            exp = mean(perc_exp),
            .groups = "drop"
            )

baseline <- df_mid %>%
  filter(year == 2020, diet_scn == "BMK") %>%
  select(region_group_IMPACT, food_group,
         sup_2020 = sup, imp_2020 = imp, exp_2020 = exp)

future <- df_mid %>%
  filter(year == 2050) %>%
  select(region_group_IMPACT, diet_scn, food_group,
         sup_2050 = sup, imp_2050 = imp, exp_2050 = exp)

change <- future %>%
  left_join(baseline, by = c("region_group_IMPACT", "food_group")) %>%
  mutate(
    sup_change = sup_2050 - sup_2020,
    imp_change = imp_2050 - imp_2020,
    exp_change = exp_2050 - exp_2020,
    region_group_IMPACT = factor(region_group_IMPACT, levels = rev(region_order)),
    food_group = factor(food_group, levels = fg_order)
  )

# Heatmap builder

vlim_global <- max(
  abs(change$sup_change),
  abs(change$imp_change),
  abs(change$exp_change),
  na.rm = TRUE
)

make_heatmap <- function(data, value_col, title) {
  
  data %>%
    ggplot(aes(y = region_group_IMPACT, x = food_group,
               fill = .data[[value_col]])) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(
      aes(
        label = sprintf("%+.1f", .data[[value_col]]),
        fontface = ifelse(abs(.data[[value_col]]) > 10, "bold", "plain"),
        colour   = ifelse(abs(.data[[value_col]]) > vlim_global * 0.55, "high", "low")
      ),
      size = 3, show.legend = FALSE
    ) +
    scale_colour_manual(values = c("high" = "white", "low" = "grey25")) +
    facet_wrap(~ diet_scn, ncol = 1) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, limits = c(-vlim_global, vlim_global),
      name = NULL
    ) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x        = element_text(size = 9, angle = 45, hjust = 1),
      axis.text.y        = element_text(size = 9),
      strip.text         = element_text(size = 10.5),
      panel.grid         = element_blank(),
      legend.position    = "bottom",
      legend.key.height  = unit(0.25, "cm"),
      legend.key.width   = unit(2.5, "cm"),
      legend.margin      = margin(t = -5, b = 0),
      plot.title         = element_text(face = "bold", size = 11)
    )
}

fig_sup <- make_heatmap(change, "sup_change", "Supply")
fig_imp <- make_heatmap(change, "imp_change", "Imports") +
  theme(axis.text.y = element_blank())
fig_exp <- make_heatmap(change, "exp_change", "Exports") +
  theme(axis.text.y = element_blank())

# combine all
bottom_label <- wrap_elements(
  textGrob(
    "Change in regional shares, 2020 to 2050 (percentage points)",
    gp = gpar(fontsize = 10.5)
  )
)

fig_all <- ((fig_sup | fig_imp | fig_exp) / bottom_label) +
  plot_layout(heights = c(1, 0.04), guides = "collect") &
  theme(plot.margin = margin(5, 8, 5, 8), legend.position = 'bottom')

print(fig_all)
ggsave("fig_data/changing_shares.png", fig_all, width = 10, height = 14, dpi = 300)







###-----------------------------------------------------------###
### DELTA SUPPLY ###
###-----------------------------------------------------------###

palette_regions <- c(
  "CNE" = "#CC79A7",   # Central & Northern Eurasia — pink
  "EAP" = "#0072B2",   # East Asia & Pacific — blue
  "EUR" = "#009E73",   # Europe — green
  "LAC" = "#E69F00",   # Latin America & Caribbean — orange
  "MEN" = "#D55E00",   # Middle East & North Africa — red-orange
  "NAM" = "#56B4E9",   # North America — light blue
  "SAS" = "#F0E442",   # South Asia — yellow
  "SSA" = "#8B4513"    # Sub-Saharan Africa — brown
)

df <- readr::read_csv("fig_data/delta_sup.csv", show_col_types = FALSE)

plot_faceted_supply <- function(df, crops, spread = "iqr", top_k = 8, ncol = 2) {
  df <- df %>%
    filter(var == "delta_supply", food_group %in% crops) %>%
    mutate(
      mean  = suppressWarnings(as.numeric(mean)) / 1000,
      iqr   = suppressWarnings(as.numeric(iqr)) / 1000,
      std   = suppressWarnings(as.numeric(std)) / 1000,
      size_2020 = suppressWarnings(as.numeric(supply_2020_mean)) / 1000, 
      region = `region_group_IMPACT`,
      country = `Region or country`,
      spread_val = .data[[spread]],
      food_group = factor(food_group, levels = crops)
    ) %>%
    filter(is.finite(mean), is.finite(spread_val), is.finite(size_2020)) 
  
  labs_df <- df %>%
    group_by(food_group) %>%
    mutate(score = abs(mean) * size_2020) %>%
    slice_max(order_by = score, n = top_k, with_ties = FALSE) %>%
    ungroup()
  
  size_breaks <- c(4, 20, 100, 500)
  
  p <- ggplot(df, aes(x = mean, y = spread_val)) +
    geom_point(aes(fill = region, size = size_2020),
               shape = 21, colour = "grey60", stroke = 0.25, alpha = 0.9) +
    geom_text_repel(data = labs_df, aes(label = country),
                    size = 4, min.segment.length = 0, seed = 123,
                    box.padding = 0.3, point.padding = 0.2, max.overlaps = 60) +
    scale_fill_manual(values = palette_regions, name = "Region") +
    scale_size(range  = c(2, 12),
               breaks = size_breaks,
               name   = "2020 supply (Mt)"
    ) +
    scale_x_continuous() +
    scale_y_continuous() +
    labs(x = "Mean change in supply, 2050–2020 (Mt)",
         y = paste0("Cross-scenario variation (", toupper(spread), ", Mt)")) +
    facet_wrap(~ food_group, scales = "free", ncol = ncol) +
    theme_light(base_size = 15) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "right",
          legend.box = "vertical",
          strip.text = element_text(colour = "black"),
          strip.background = element_rect(fill = "grey90", colour = NA))
  
  return(p)
}

ggsave("fig_data/delta_sup.png", 
       plot_faceted_supply(df, crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                         'Legumes, nuts and seeds', 'Oilcrops and sugar crops')), 
       width = 13, height = 13, units = "in", dpi = 300)





###-----------------------------------------------------------###
### DECOMPOSITION ###
###-----------------------------------------------------------###

reg_df <- readr::read_csv("fig_data/decompose.csv", show_col_types = FALSE)

plot_crop_variance_stacks <- function(reg_df, crop_code = "Grains", vari = "supply") {
  
  d <- reg_df %>%
    mutate(
      outcome = factor(y,
                       levels = c("supply", "export", "demand", "import",
                                  "delta_supply", "delta_export", "delta_demand", "delta_import"),
                       labels = c("Supply", "Export", "Demand", "Import",
                                  "DeltaSupply", "DeltaExport", "DeltaDemand", "DeltaImport")),
      # scale mean and variance for nicer numbers
      mean_k = mean / 1e3,      
      var_M  = var  / 1e6,
      region = fct_rev(factor(region_group_IMPACT))
    ) %>%
    filter(
      food_group == crop_code,
           year %in% c(2030, 2050),
           y == vari) %>%
    # long for stacked bars
    pivot_longer(c(diet_share, lib_share, RCP_share),
                 names_to = "lever", values_to = "share") %>%
    mutate(
      lever = factor(recode(lever,
                            diet_share="Diet",
                            lib_share ="Trade",
                            RCP_share ="Climate"),
                     levels = c("Diet","Trade","Climate")),
      var_comp_M = share * var_M      # component = share × total variance
    )
  
  # mean labels at the end of each total bar
  lab_df <- d %>%
    distinct(year, food_group, region, var_M, mean_k)
  
  ggplot(d, aes(x = var_comp_M, y = region, fill = lever)) +
    geom_col(width = 0.8) +
    # mean (scaled to 'k') shown just beyond bar end
    geom_text(data = lab_df,
              aes(x = var_M , y = region,
                  label = paste0(" ", number(mean_k, accuracy = 1))),
              inherit.aes = FALSE, hjust = 0, size = 4) +
    facet_wrap(~ year, ncol = 1, scales = "free") +
    scale_y_reordered() +
    scale_x_continuous(labels = scales::label_number(big.mark = ","),
                       expand  = expansion(mult = c(0, 0.25))) +  # room for μ labels
    scale_fill_manual(values = c("Diet"="#0072B2", "Trade"="#009E73", "Climate"="#E69F00"), 
                      name = NULL) +
    labs(
      subtitle = crop_code,
      x = "Variance across scenarios (Mt²)", y = NULL, fill = "Lever"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      panel.grid.major.y = element_blank()
    ) 
}


# combine plots
crop_order <- c("Grains",
                "Roots and tubers",
                "Fruits and vegetables",
                "Soybeans",
                "Legumes, nuts and seeds",
                "Oilcrops and sugar crops")

# helpers
hide_x_title <- function(p) p + theme(axis.title.x = element_blank())
show_x_title <- function(p) p + theme(axis.title.x = element_text())

hide_y <- function(p) p + theme(axis.title.y = element_blank(),
                                axis.text.y  = element_blank(),
                                axis.ticks.y = element_blank())
gap <- 15 

##### supply
plots <- lapply(crop_order, \(g) plot_crop_variance_stacks(reg_df, crop_code = g, vari = "supply"))
plots <- lapply(plots, \(p) p + theme(plot.margin = margin(gap, gap, gap, gap)))

# layout bookkeeping
n     <- length(plots)
ncol  <- 3
nrow  <- ceiling(n / ncol)

# indices by rows (row-major fill)
row_indices <- split(seq_len(n), ceiling(seq_len(n) / ncol))

# choose which subplot keeps the x-axis *title*: center of the last row
last_row     <- row_indices[[length(row_indices)]]
keep_xtitle  <- last_row[ceiling(length(last_row) / 2)]

# leftmost plot in each row keeps the y-axis labels
leftmost_per_row <- vapply(row_indices, function(v) v[1], integer(1))

# apply visibility rules
plots_ax <- lapply(seq_along(plots), function(i) {
  p <- plots[[i]]
  # x: keep tick labels everywhere; only hide/show the title
  p <- if (i == keep_xtitle) show_x_title(p) else hide_x_title(p)
  # y: only leftmost in each row shows labels/ticks
  if (!(i %in% leftmost_per_row)) p <- hide_y(p)
  p
})

# assemble and collect legend
final_plot <- wrap_plots(plots_ax, ncol = ncol, guides = "collect") &
  theme(legend.position = "bottom")

final_plot
ggsave("fig_data/decompose_supply.png", final_plot, width = 12, height = 12, dpi = 300)

##### exports
plots <- lapply(crop_order, \(g) plot_crop_variance_stacks(reg_df, crop_code = g, vari = "export"))
plots <- lapply(plots, \(p) p + theme(plot.margin = margin(gap, gap, gap, gap)))

# layout bookkeeping
n     <- length(plots)
ncol  <- 3
nrow  <- ceiling(n / ncol)

# indices by rows (row-major fill)
row_indices <- split(seq_len(n), ceiling(seq_len(n) / ncol))

# choose which subplot keeps the x-axis *title*: center of the last row
last_row     <- row_indices[[length(row_indices)]]
keep_xtitle  <- last_row[ceiling(length(last_row) / 2)]

# leftmost plot in each row keeps the y-axis labels
leftmost_per_row <- vapply(row_indices, function(v) v[1], integer(1))

# apply visibility rules
plots_ax <- lapply(seq_along(plots), function(i) {
  p <- plots[[i]]
  # x: keep tick labels everywhere; only hide/show the title
  p <- if (i == keep_xtitle) show_x_title(p) else hide_x_title(p)
  # y: only leftmost in each row shows labels/ticks
  if (!(i %in% leftmost_per_row)) p <- hide_y(p)
  p
})

# assemble and collect legend
final_plot <- wrap_plots(plots_ax, ncol = ncol, guides = "collect") &
  theme(legend.position = "bottom")

final_plot
ggsave("fig_data/decompose_export.png", final_plot, width = 12, height = 12, dpi = 300)








###-----------------------------------------------------------###
### RESILIENCE ###
###-----------------------------------------------------------###

agg_df <- readr::read_csv("fig_data/hhi_sum.csv", show_col_types = FALSE)

prep_structured <- function(df) {
  df %>%
    mutate(
      year2020 = scen == "2020" | diet_scn == "2020",
      AID_pct  = (if (max(df$import_dep, na.rm = TRUE) <= 1.5) 100 else 1) * import_dep,
      HHI      = hhi,
      diet_scn = as.character(diet_scn),
      lib_scn  = as.character(lib_scn),
      RCP      = as.character(RCP),
      lib_scn = recode(lib_scn, "low" = "high", "high" = "low")
    )
}

plot_tradeoff_structured <- function(df, crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                                   'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
                                     diet_levels = c("BMK","FLX","PSC","VEG","VGN","2020"),
                                     lib_levels = c("high", "low", "2020"),
                                     rcp_keep  = "2.6") {
  
  diet_pal <- c(BMK="#CC3311", FLX="#EE7733", PSC="#0077BB", VEG="#33BBEE", VGN="#117733", `2020`="black")
  
  d <- prep_structured(df) %>% 
    filter(food_group %in% crops,
           RCP %in% c("2020", rcp_keep))
  
  # Baseline points (one per crop)
  base <- d %>% filter(year2020) %>%
    group_by(food_group) %>%
    summarise(AID_pct = first(AID_pct), HHI = first(HHI), .groups="drop") %>%
    mutate(food_group = factor(food_group, levels = crops),
           baseline_key = "2020 baseline")
  
  
  d_plot <- d %>% filter(!year2020) %>%
    mutate(
      diet_scn = factor(diet_scn, levels = diet_levels),
      lib_scn = factor(lib_scn, levels = lib_levels),
      food_group = factor(food_group, levels = crops),
    )
  
  ggplot(d_plot, aes(AID_pct, HHI, colour = diet_scn, shape = lib_scn)) +
    geom_hline(yintercept = 0.25, linetype = "dashed", colour = "grey70") +
    geom_vline(xintercept = 50,  linetype = "dashed", colour = "grey70") +
    geom_point(size = 3, alpha = 0.9) +
    geom_point(data = base, aes(AID_pct, HHI, alpha = baseline_key), inherit.aes = FALSE,
               shape = 4, size = 4, stroke = 1.1, colour = "black") +
    facet_wrap(~ food_group, ncol = 2, scales = "free") +
    scale_x_continuous(limits = c(0, 100), labels = label_number(accuracy = 1, suffix = "%")) +
    scale_y_continuous(limits = c(0, 1),   labels = label_number(accuracy = 0.01)) +
    scale_colour_manual(values = diet_pal, breaks = c("BMK","FLX","PSC","VEG","VGN")) +
    scale_shape_manual(values = c("high" = 16, "low" = 17, `2020` = 4), 
                       breaks = c("high", "low")) +
    scale_alpha_manual(
      name   = "",
      values = c("2020 baseline" = 1),
      breaks = "2020 baseline",
      guide  = guide_legend(override.aes = list(shape = 4, colour = "black", size = 4))
    ) +
    guides(
      colour = guide_legend(order = 1),
      shape  = guide_legend(order = 2),
      alpha  = guide_legend(order = 3)
    ) +
    labs(x = "Import dependence (% of demand)",
         y = "HHI over Imports, weighted",
         colour = "Diet", shape = "Trade constraints") +
    theme_minimal(base_size = 16) +
    theme(panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"),
          panel.spacing.x = unit(10, "mm"),  # horizontal gap between facet columns
          panel.spacing.y = unit(6,  "mm"),  # vertical gap between facet rows
          plot.margin     = margin(10, 14, 10, 10))
}

p <- plot_tradeoff_structured(agg_df)
print(p)
ggsave("fig_data/hhi_sum_2.6.png", p, 
       width = 10, height = 8, dpi = 300)

p <- plot_tradeoff_structured(agg_df, rcp_keep  = "7")
print(p)
ggsave("fig_data/hhi_sum_7.0.png", p, 
       width = 10, height = 8, dpi = 300)








agg_df_reg <- readr::read_csv("fig_data/hhi_sum_reg.csv", show_col_types = FALSE)

prep_structured <- function(df) {
  df %>%
    mutate(
      year2020 = scen == "2020" | diet_scn == "2020",
      AID_pct  = (if (max(df$import_dep, na.rm = TRUE) <= 1.5) 100 else 1) * import_dep,
      HHI      = hhi,
      diet_scn = as.character(diet_scn),
      lib_scn  = as.character(lib_scn),
      RCP      = as.character(RCP),
      lib_scn = recode(lib_scn, "low" = "high", "high" = "low")
    )
}

plot_tradeoff_structured <- function(
    df,
    crops        = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                     'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
    diet_levels  = c("BMK","FLX","PSC","VEG","VGN","2020"),
    lib_levels = c("high", "low", "2020"),
    rcp_keep     = "2.6",   # 
    region_levels = c("CNE", "EAP", "EUR", "LAC", "MEN", "NAM", "SAS", "SSA")  
) {
  
  diet_pal <- c(BMK="#CC3311", FLX="#EE7733", PSC="#0077BB",
                VEG="#33BBEE", VGN="#117733", `2020`="black")
  
  d <- prep_structured(df) %>% 
    filter(food_group %in% crops,
           RCP %in% c("2020", rcp_keep)) %>%    
    mutate(region = `region_group_IMPACT`)
  
  # baseline, per food_group and region 
  base <- d %>%
    filter(year2020) %>%
    group_by(food_group, region) %>%        
    summarise(AID_pct = first(AID_pct),
              HHI     = first(HHI),
              .groups = "drop") %>%
    mutate(
      food_group = factor(food_group, levels = crops),
      region     = factor(region, levels = region_levels),
      baseline_key = "2020 baseline"
    )
  
  # plotting data (non-baseline)
  d_plot <- d %>%
    filter(!year2020) %>%
    mutate(
      diet_scn   = factor(diet_scn, levels = diet_levels),
      lib_scn    = factor(lib_scn, levels = lib_levels),
      food_group = factor(food_group, levels = crops),
      region     = factor(region, levels = region_levels)
    )
  
  ggplot(d_plot, aes(AID_pct, HHI, colour = diet_scn, shape = lib_scn)) +
    geom_hline(yintercept = 0.25, linetype = "dashed", colour = "grey70") +
    geom_vline(xintercept = 50,  linetype = "dashed", colour = "grey70") +
    geom_point(size = 3, alpha = 0.9) +
    geom_point(
      data = base,
      aes(AID_pct, HHI, alpha = baseline_key),
      inherit.aes = FALSE,
      shape = 4, size = 4, stroke = 1.1, colour = "black"
    ) +
    # facet by food_group (rows) and region (columns)
  facet_grid(
    cols = vars(food_group),
    rows = vars(region),
    scales = "free_y",
    labeller = labeller(food_group = label_wrap_gen(width = 16),
                        region = label_wrap_gen(width = 20))
  ) +
    scale_x_continuous(
      limits = c(0, 100),
      labels = label_number(accuracy = 1, suffix = "%")
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      labels = label_number(accuracy = 0.01)
    ) +
    scale_colour_manual(
      values = diet_pal,
      breaks = c("BMK","FLX","PSC","VEG","VGN")
    ) +
    scale_shape_manual(
      values = c("high" = 16, "low" = 17, `2020` = 4),
      breaks = c("high", "low")
    ) +
    scale_alpha_manual(
      name   = "",
      values = c("2020 baseline" = 1),
      breaks = "2020 baseline",
      guide  = guide_legend(
        override.aes = list(shape = 4, colour = "black", size = 4)
      )
    ) +
    guides(
      colour = guide_legend(order = 1),
      shape  = guide_legend(order = 2),
      alpha  = guide_legend(order = 3)
    ) +
    labs(
      x = "Import dependence (% of demand)",
      y = "HHI over Imports, weighted",
      colour = "Diet",
      shape = "Trade constraints"
    ) +
    theme_minimal(base_size = 16) +
    theme(
      panel.grid.minor = element_blank(),
      strip.text       = element_text(face = "bold"),
      panel.spacing.x  = unit(10, "mm"),
      panel.spacing.y  = unit(6,  "mm"),
      plot.margin      = margin(10, 14, 10, 10)
    )
}

p <- plot_tradeoff_structured(agg_df_reg)
print(p)
ggsave("fig_data/hhi_sum_reg_2.6.png", p, 
       width = 18, height = 17, dpi = 300)