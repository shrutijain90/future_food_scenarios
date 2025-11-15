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
  
  p <- ggplot(df, aes(x = mean, y = spread_val)) +
    geom_point(aes(fill = region, size = size_2020),
               shape = 21, colour = "grey60", stroke = 0.25, alpha = 0.9) +
    geom_text_repel(data = labs_df, aes(label = country),
                    size = 4, min.segment.length = 0, seed = 123,
                    box.padding = 0.3, point.padding = 0.2, max.overlaps = 60) +
    scale_fill_manual(values = palette_regions, name = "Region") +
    scale_size(range  = c(2, 12),    
               breaks = scales::breaks_pretty(n = 4)(range(df$size_2020, na.rm = TRUE)),
               name   = "2020 supply (Mt)"
    ) +
    scale_x_continuous() +
    scale_y_continuous() +
    labs(x = "Mean change in supply, 2050–2020 (Mt)",
         y = paste0("Across-scenario uncertainty (", toupper(spread), ", Mt)")) +
    facet_wrap(~ food_group, scales = "free", ncol = ncol) +
    theme_light(base_size = 17) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "right",
          legend.box = "vertical",
          strip.text = element_text(colour = "black"),
          strip.background = element_rect(fill = "grey90", colour = NA))
  
  return(p)
}

p <- plot_faceted_supply(df, crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                       'Legumes, nuts and seeds', 'Oilcrops and sugar crops'))
print(p)
ggsave("fig_data/delta_sup.png", p, 
       width = 14, height = 16, units = "in", dpi = 300)



### decomposition ###

reg_df <- readr::read_csv("fig_data/decompose.csv", show_col_types = FALSE)

plot_crop_variance_stacks <- function(reg_df, crop_code = "Grains") {
  
  d <- reg_df %>%
    filter(food_group == crop_code,
           y %in% c("delta_supply","delta_export")) %>%
    mutate(
      outcome = factor(y,
                       levels = c("delta_supply","delta_export"),
                       labels = c("Supply","Export")),
      # scale mean and variance for nicer numbers
      mean_k = mean_delta / 1e3,      
      var_M  = var_delta  / 1e6,
      region = `Region Name`
    ) %>%
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
    distinct(outcome, region, var_M, mean_k)
  
  ggplot(d, aes(x = var_comp_M, y = region, fill = lever)) +
    geom_col(width = 0.8) +
    # mean (scaled to 'k') shown just beyond bar end
    geom_text(data = lab_df,
              aes(x = var_M , y = region,
                  label = paste0(" ", number(mean_k, accuracy = 1))),
              inherit.aes = FALSE, hjust = 0, size = 4) +
    facet_wrap(~ outcome, ncol = 1) +
    scale_y_reordered() +
    scale_x_continuous(labels = scales::label_number(big.mark = ","),
                       expand  = expansion(mult = c(0, 0.15))) +  # room for μ labels
    scale_fill_manual(values = c("Diet"="#0072B2", "Liberalization"="#009E73", "RCP"="#E69F00"), 
                      name = NULL) +
    labs(
      subtitle = crop_code,
      x = "Variance across scenarios (Mt²)", y = NULL, fill = "Lever"
    ) +
    theme_minimal(base_size = 16) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      panel.grid.major.y = element_blank()
    )
}

p <- plot_crop_variance_stacks(reg_df, crop_code = "Grains")  
print(p)
ggsave("fig_data/decompose_grains.png", plot_crop_variance_stacks(reg_df, crop_code = "Grains"), 
       width = 8, height = 8, dpi = 300)
ggsave("fig_data/decompose_roots.png", plot_crop_variance_stacks(reg_df, crop_code = "Roots and tubers"), 
       width = 8, height = 8, dpi = 300)
ggsave("fig_data/decompose_fnv.png", plot_crop_variance_stacks(reg_df, crop_code = "Fruits and vegetables"), 
       width = 8, height = 8, dpi = 300)
ggsave("fig_data/decompose_soybean.png", plot_crop_variance_stacks(reg_df, crop_code = "Soybeans"), 
       width = 8, height = 8, dpi = 300)
ggsave("fig_data/decompose_legumes.png", plot_crop_variance_stacks(reg_df, crop_code = "Legumes, nuts and seeds"), 
       width = 8, height = 8, dpi = 300)
ggsave("fig_data/decompose_oilsug.png", plot_crop_variance_stacks(reg_df, crop_code = "Oilcrops and sugar crops"), 
       width = 8, height = 8, dpi = 300)


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
                                     rcp_levels  = c("2.6","7")) {
  
  diet_pal <- c(BMK="#CC3311", FLX="#EE7733", PSC="#0077BB", VEG="#33BBEE", VGN="#117733", `2020`="black")
  
  d <- prep_structured(df) %>% filter(food_group %in% crops)
  
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
      RCP  = factor(RCP, levels = rcp_levels, 
                        labels = c("RCP 2.6","RCP 7.0")),
      food_group = factor(food_group, levels = crops),
    )
  
  ggplot(d_plot, aes(AID_pct, HHI, colour = diet_scn, shape = lib_scn)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey70") +
    geom_vline(xintercept = 50,  linetype = "dashed", colour = "grey70") +
    geom_point(size = 3, alpha = 0.9) +
    geom_point(data = base, aes(AID_pct, HHI, alpha = baseline_key), inherit.aes = FALSE,
               shape = 4, size = 4, stroke = 1.1, colour = "black") +
    facet_grid(rows = vars(food_group), cols = vars(RCP), scales = "free_y",
               labeller = labeller(food_group = label_wrap_gen(width = 16))) +
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
       width = 12, height = 10, dpi = 300)


plot_tradeoff_structured <- function(df, crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                                   'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
                                     diet_levels = c("BMK","FLX","PSC","VEG","VGN","2020"),
                                     lib_levels  = c("low","high","2020"),
                                     rcp_levels  = c("2.6","7")) {
  
  d <- prep_structured(df) %>% dplyr::filter(food_group %in% crops) 
  
  # baseline risk per crop (2020), replicated across RCP facets
  base <- d %>%
    dplyr::filter(year2020) %>%
    dplyr::group_by(food_group) %>%
    dplyr::summarise(risk = dplyr::first(risk), .groups = "drop") %>%
    dplyr::mutate(
      food_group = factor(food_group, levels = crops),
    ) 
  
  # plotting data (non-baseline) with fixed orders and facet headers
  d_plot <- d %>%
    dplyr::filter(!year2020) %>%
    dplyr::mutate(
      diet_scn    = factor(diet_scn, levels = diet_levels),
      lib_scn     = factor(lib_scn,  levels = lib_levels),
      food_group = factor(food_group, levels = crops),
      RCP         = factor(RCP, levels = rcp_levels,
                           labels = c("RCP 2.6","RCP 7.0"))
    )
  
  pos <- position_dodge(width = 0.70)
  
  ggplot(d_plot, aes(x = diet_scn, y = risk, fill = lib_scn)) +
    # dashed 2020 baseline in every facet
    geom_hline(data = base, aes(yintercept = risk), inherit.aes = FALSE,
               colour = "grey60", linetype = "dashed") +
    geom_col(position = pos, width = 0.62) +
    facet_grid(rows = vars(food_group), cols = vars(RCP),
               labeller = labeller(food_group = label_wrap_gen(width = 16))) +
    scale_y_continuous(labels = scales::label_percent(accuracy = 1),
                       expand  = expansion(mult = c(0, 0.08))) +
    labs(x = "Diet scenario", y = "Demand at risk", fill = "Liberalization") +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      panel.spacing.x = grid::unit(10, "mm"),
      panel.spacing.y = grid::unit(6,  "mm"),
      plot.margin     = margin(10, 14, 10, 10)
    )
}

p <- plot_tradeoff_structured(agg_df)
print(p)
ggsave("fig_data/hhi_risk.png", p, 
       width = 10, height = 9, dpi = 300)



country_df <- readr::read_csv("fig_data/hhi.csv", show_col_types = FALSE)

plot_country_arrows <- function(df_ctry,
                                crops = c('Grains', 'Roots and tubers', 'Fruits and vegetables', 'Soybeans',
                                          'Legumes, nuts and seeds', 'Oilcrops and sugar crops'),
                                scen_choice = "2050, BMK, high, 7.0",
                                top_mass = 0.80,   # keep countries covering X% of imports (per crop)
                                x_thr = 50, y_thr = 0.5) {
  
  d <- df_ctry %>%
    filter(food_group %in% crops) %>%
    select(food_group, abbreviation, scen, perc_imp, hhi, import) %>%
    drop_na(perc_imp, hhi, import) %>% 
    dplyr::mutate(import = import / 1e3)
  
  # put perc_imp on 0–100 if it’s 0–1
  if (max(d$perc_imp, na.rm = TRUE) <= 1.5) d$perc_imp <- d$perc_imp * 100
  
  base <- d %>% filter(scen == "2020") %>%
    rename(x0 = perc_imp, y0 = hhi, import0 = import)
  
  sc   <- d %>% filter(scen == scen_choice) %>%
    rename(x1 = perc_imp, y1 = hhi, import1 = import)
  
  dd <- inner_join(base, sc, by = c("food_group","abbreviation")) %>%
    group_by(food_group) %>%
    arrange(desc(import1), .by_group = TRUE) %>%
    mutate(cum = cumsum(import1)/sum(import1, na.rm=TRUE)) %>%
    filter(cum <= top_mass) %>%
    ungroup() %>%
    mutate(
      risk0 = (x0/100)*y0, risk1 = (x1/100)*y1,
      drisk = risk1 - risk0,
      food_group = factor(food_group, levels = crops)
    )
  
  ggplot(dd) +
    geom_hline(yintercept = y_thr, linetype = "dashed", colour = "grey70") +
    geom_vline(xintercept = x_thr, linetype = "dashed", colour = "grey70") +
    geom_segment(aes(x = x0, y = y0, xend = x1, yend = y1, colour = drisk),
                 arrow = arrow(length = unit(2.5, "mm"), type = "closed"),
                 linewidth = 0.5, alpha = 0.9) +
    geom_point(aes(x = x1, y = y1, size = import1), shape = 21,
               fill = "white", colour = "black", stroke = 0.4) +
    ggrepel::geom_text_repel(aes(x = x1, y = y1, label = abbreviation),
                             size = 3, max.overlaps = 12, min.segment.length = 0) +
    facet_wrap(~ food_group, ncol = 2,
               labeller = labeller(food_group = label_wrap_gen(width = 16))) +
    scale_colour_gradient2(low = "#2c7bb6", mid = "grey50", high = "#d7191c",
                           midpoint = 0, name = "Δ (imp%×HHI)") +
    scale_size_continuous(range = c(1.8, 7), name = "2050 Imports (Mt)") +
    scale_x_continuous(limits = c(0,100), labels = label_number(suffix = "%")) +
    scale_y_continuous(limits = c(0,1),   labels = label_number(accuracy = 0.1)) +
    labs(title = paste("2020 →", scen_choice),
         x = "Import dependence (% of country demand)", y = "HHI over country imports") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"),
          panel.spacing.y = unit(6, "mm"))
}


p <- plot_country_arrows(country_df, scen_choice="2050, BMK, low, 7.0", top_mass=0.8)
print(p)
ggsave("fig_data/hhi_country.png", p, 
       width = 10, height = 7, dpi = 300)