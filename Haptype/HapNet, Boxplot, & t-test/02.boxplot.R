suppressMessages(library(ggpubr))
suppressMessages(library(gghalves))
suppressMessages(library(ggplot2))
suppressMessages(library(pegas))
suppressMessages(library(vcfR))
suppressMessages(library(cowplot))
suppressMessages(library(randomcoloR))

args <- commandArgs(trailingOnly = TRUE)
# args[1]: input vcf file
# args[2]: title in plot (Gene ID)
# args[3]: file name of hapnet
# args[4]: phenotype file (Accession\tHaplotype\tPhenotype\n)
# args[5]: gene info
# args[6]: file name of boxplot + hapnet
# args[7]: scale

# Colors
palette <- c("#6996E7", "#65EC9F", "#64669E", "#B7B3E7", "#E66788", "#E7DF99", "#D4A748", "#57B078", "#B0E6DC", "#648C86", "#E740B4", "#7DE84F", "#D541EC", "#B4DD6F", "#CE71DF", "#6C6BE6", "#E49E94", "#D0C1D4", "#8DD3EA", "#B19F76", "#9D5381", "#6EAAD3", "#E3E44A", "#ECA8D0", "#E17545", "#6F36D2", "#D795E1", "#5DDFD1", "#E3E3D2", "#B0EAB4")

#################################### HapNet ####################################
vcfR_hn_orivcf <- read.vcfR(args[1]) # Load vcf file

dna <- vcfR2DNAbin(vcfR_hn_orivcf, consensus = T, extract.haps = F)
hap <- haplotype(dna)
hap <- sort(hap, what = "labels")
hap.net <- haploNet(hap, getProb = FALSE)

hap_order <- attr(hap, "dimnames")[[1]]


pal <- palette[1:length(hap_order)]
names(pal) <- hap_order

sample_order <- attr(dna, "dimnames")[[1]]
sample_hap_order <- c()
detail_lis <- c()
for (i in seq_along(hap_order)){
    sample_hap_order <- c(sample_hap_order, rep(hap_order[i], length(attr(hap, "index")[[i]])))
    detail_lis <- c(detail_lis, sample_order[attr(hap, "index")[[i]]])
}
names(sample_hap_order) <- detail_lis
groupf <- data.frame(Sample = sample_order, Group = sample_hap_order[sample_order])  # length(labels(dna)) == 740
ind.hap<-as.data.frame(with(
    stack(setNames(attr(hap, "index"), rownames(hap))), 
    table(hap=ind, Sample=rownames(dna)[values])
))

ind.hap2=ind.hap[ind.hap$Freq==1,c(2,1)]
res1=merge(ind.hap2,groupf)
hap.pies <- with(
    stack(setNames(attr(hap,'index'),1:length(attr(hap,'index')))),
    table(hap=as.numeric(as.character(ind)),groupf[values,"Group"])
)


# par(mar=c(9,9,9,9))
message(args[2])
pdf(args[3], width=5, height=5)
dev.control(displaylist="enable")
plot(hap.net, size=attr(hap.net, "freq")^0.5, bg=pal, pie=hap.pies, scale.ratio = 1, fast = F)
# plot(c(1,2,3), c(1,2,3))
HapNet <- recordPlot()
dev.off()


#################################### Box Plot ####################################
data <- read.table(args[4], header = T, stringsAsFactors = F) # "plot.boxplot.txt"

FreqThreshold <- 30

Haps <- unique(data$Haplotype)
PhenoMean <- c()
ValidHaps <- c()
for(Hap in Haps){
    if(length(which(data$Haplotype == Hap)) > FreqThreshold){
        ValidHaps <- c(ValidHaps, Hap)
        PhenoMean <- c(PhenoMean, median(data$Phenotype[which(data$Haplotype == Hap)]))
    }
}
HapsOrder <- ValidHaps[order(-PhenoMean)]
data <- data[which(data$Haplotype %in% ValidHaps),]
data$Haplotype <- factor(data$Haplotype, levels = HapsOrder) 

my_comparisons <- list()
for (x in seq(1, length(HapsOrder)-1, 2)) {
    my_comparisons <- append(my_comparisons, list(c(HapsOrder[x], HapsOrder[x+1])))
}
if(length(HapsOrder) > 2){
    for (x in seq(2, length(HapsOrder)-1, 2)) {
        my_comparisons <- append(my_comparisons, list(c(HapsOrder[x], HapsOrder[x+1])))
    }
}

# print(pal)
HapPhe <- ggplot(data,aes(x=Haplotype,y=Phenotype,fill=Haplotype,colour=Haplotype))+
    geom_half_boxplot(width=0.8,col="black")+
    #geom_half_violin()+
    geom_half_point(size=1) +
    scale_fill_manual(values=pal) + 
    scale_colour_manual(values=pal) + 
    stat_compare_means(comparisons = my_comparisons, method = "t.test") + theme_classic()

scale = as.numeric(args[7])

Combine <- plot_grid(
    HapNet,
    HapPhe,
    ncol = 2,
    rel_widths = c(10, 9*scale)
)
Title <- ggplot() + annotate("text", x = 1, y = 3,size = 4*scale, label = paste(args[2], "\n", args[5], sep="")) + theme_void()

pdf(args[6], width=9*scale, height=8)
plot_grid(
    Title,
    Combine,
    nrow = 2,
    rel_heights = c(4, 10)
)
dev.off()