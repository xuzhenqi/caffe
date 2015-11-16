function feat = load_features_from_file(filename)
	fea1=sprintf(filename);
	feafile=fopen(fea1,'rb');
	[dims]=fread(feafile,1,'int');
	[num]=fread(feafile,1,'int');
	feat=fread(feafile,dims*num,'float');
	fclose(feafile);
	feat = reshape(features,dims,num);
end
