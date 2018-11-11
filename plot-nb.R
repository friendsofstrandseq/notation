#!/usr/bin/env Rscript
library(ggplot2)

p = 0.4
x = seq(0:20)

y1 = dnbinom(x, size=0.125*(1-p)/p, prob=p)
y2 = dnbinom(x, size=3*(1-p)/p, prob=p)
y3 = dnbinom(x, size=6*(1-p)/p, prob=p)

df = data.frame(x,y1,y2,y3)

ggplot(df, aes(x)) +
	geom_line(aes(y = y1), , color = "#f4b016", size = 1) +
	geom_point(aes(y = y1), color = "#f4b016", size = 3) + 
	geom_line(aes(y = y2), color = "#1657f4", size = 1) +
	geom_point(aes(y = y2), color = "#1657f4", size = 3) +
	geom_line(aes(y = y3), color = "#f4166d", size = 1) +
	geom_point(aes(y = y3), color = "#f4166d", size = 3) +
	theme_bw() +
	theme(panel.border = element_blank(), panel.grid.major = element_blank(),
panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggsave('nb.pdf')
