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

setwd('/Users/shrutijain/Library/CloudStorage/OneDrive-Nexus365/DPhil/OPSIS/Data/')

### delta supply ###

palette_regions <- c(
  "Africa"="#E69F00","Asia"="#0072B2","Europe"="#009E73",
  "Latin America and the Caribbean"="#F0E442",
  "Northern America"="#56B4E9","Oceania"="#CC79A7"
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
      region = `Region Name`,
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
         y = paste0("Across-scenario uncertainty (", toupper(spread), ", Mt)")) +
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



### decomposition ###

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
      region = `Region Name`
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
                            lib_share ="Liberalization",
                            RCP_share ="RCP"),
                     levels = c("Diet","Liberalization","RCP")),
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
    scale_fill_manual(values = c("Diet"="#0072B2", "Liberalization"="#009E73", "RCP"="#E69F00"), 
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
ggsave("fig_data/decompose_supply.png", final_plot, width = 14, height = 11, dpi = 300)

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
ggsave("fig_data/decompose_export.png", final_plot, width = 14, height = 11, dpi = 300)


### resilience ###

agg_df <- readr::read_csv("fig_data/hhi_sum.csv", show_col_types = FALSE)

prep_structured <- function(df) {
  df %>%
    mutate(
      year2020 = scen == "2020" | diet_scn == "2020",
      AID_pct  = (if (max(df$import_dep, na.rm = TRUE) <= 1.5) 100 else 1) * import_dep,
      HHI      = hhi,
      diet_scn = as.character(diet_scn),
      lib_scn  = as.character(lib_scn),
      RCP      = as.character(RCP)
    )
}

plot_tradeoff_structured <- function(df, crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                                   'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
                                     diet_levels = c("BMK","FLX","PSC","VEG","VGN","2020"),
                                     lib_levels  = c("low","high","2020"),
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
    scale_shape_manual(values = c(low = 16, high = 17, `2020` = 4), 
                       breaks = c("low","high")) +
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
         colour = "Diet", shape = "Liberalization") +
    theme_minimal(base_size = 16) +
    theme(panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"),
          panel.spacing.x = unit(10, "mm"),  # horizontal gap between facet columns
          panel.spacing.y = unit(6,  "mm"),  # vertical gap between facet rows
          plot.margin     = margin(10, 14, 10, 10))
}

p <- plot_tradeoff_structured(agg_df)
print(p)
ggsave("fig_data/hhi_sum.png", p, 
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
      RCP      = as.character(RCP)
    )
}

plot_tradeoff_structured <- function(
    df,
    crops        = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                     'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
    diet_levels  = c("BMK","FLX","PSC","VEG","VGN","2020"),
    lib_levels   = c("low","high","2020"),
    rcp_keep     = "2.6",   # 
    region_levels = c("Africa","Asia","Europe","Latin America and the Caribbean",
                      "Oceania","Northern America")  
) {
  
  diet_pal <- c(BMK="#CC3311", FLX="#EE7733", PSC="#0077BB",
                VEG="#33BBEE", VGN="#117733", `2020`="black")
  
  d <- prep_structured(df) %>% 
    filter(food_group %in% crops,
           RCP %in% c("2020", rcp_keep)) %>%    
    mutate(region = `Region Name`)
  
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
    rows = vars(food_group),
    cols = vars(region),
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
      values = c(low = 16, high = 17, `2020` = 4),
      breaks = c("low","high")
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
      shape  = "Liberalization"
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
ggsave("fig_data/hhi_sum_reg.png", p, 
       width = 18, height = 12, dpi = 300)