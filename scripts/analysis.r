library(lidR)

#TLS data is the de heaviest, we need to read the catalog and decimate heavy
ctg1<- readLAScatalog("//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/tiles")
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/decimated/"
dir.create(outdir)
plot(ctg1)
opt_output_files(ctg1) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
cleaned_ctg<- decimate_points(ctg1, random(30000))

###TLS classify the points
# classify ground pointss
outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/classified/"
dir.create(outdir)
opt_chunk_buffer(cleaned_ctg)<- 2
opt_chunk_size(cleaned_ctg)<-20
plot(cleaned_ctg,chunk=TRUE)
opt_output_files(cleaned_ctg) <- paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
classified_ctg <- classify_ground(cleaned_ctg, csf(cloth_resolution=0.3, rigidness=3L))

outdir<-"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/TLS/plot1/ground/"
dir.create(outdir)
opt_output_files(classified_ctg)<-paste0(outdir,"/retile_{XLEFT}_{YBOTTOM}")
dtm<- rasterize_terrain(classified_ctg, 0.2, tin(), pkg="terra")


## TLS create a DTM in disk
dtm1<- rasterize_terrain(ground_ctg, 0.2, tin(), pkg="terra")


