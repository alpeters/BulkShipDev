load("data/bulkers_WFR.Rda")

# Look for relevant columns
colnames(bulkers_df)[str_detect(colnames(bulkers_df), regex('engine', ignore_case = TRUE))]
colnames(bulkers_df)[str_detect(colnames(bulkers_df), regex('fuel', ignore_case = TRUE))]

# Plot frequency of engine speeds
bulkers_df %>% 
  mutate(RPM = str_extract(Main.Engine.Detail, '[[:digit:]]*(?=rpm|RPM)') %>% 
           as.numeric()) %>% 
  # group_by(RPM > 300) %>% 
  # summarise(n())$
  ggplot(aes(x = RPM)) +
  geom_histogram()

# Plot frequency of fuel types
bulkers_df %>% 
  ggplot(aes(x = Main.Engine.Fuel.Type)) +
  geom_bar()
  
