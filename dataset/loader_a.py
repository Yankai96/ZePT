def get_loader_m(args):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_train.json')
        data = json.load(open(json_path))
        
        for each in data:
            name = each["label"].split('.')[0]
            train_img.append(os.path.join(args.data_root_path, each["img"]))
            train_lbl.append(os.path.join(args.data_root_path, each["label"]))
            train_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            train_name.append(name)
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))


    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_val.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["label"].split('.')[0]
            val_img.append(os.path.join(args.data_root_path, each["img"]))
            val_lbl.append(os.path.join(args.data_root_path, each["label"]))
            val_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))


    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_test.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["label"].split('.')[0]
            test_img.append(os.path.join(args.data_root_path, each["img"]))
            test_lbl.append(os.path.join(args.data_root_path, each["label"]))
            test_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    if args.phase == 'train':
        data_part = partition_dataset(
            data=data_dicts_train,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=True,
        )[dist.get_rank()]
        print(len(data_part))
        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformSmartCacheDataset(data=data_part, transform=train_transforms, datasetkey=args.datasetkey, cache_rate=args.cache_rate, num_init_workers=8, num_replace_workers=8)
            else:
                train_dataset = CacheDataset(data=data_part, transform=train_transforms, cache_rate=args.cache_rate, num_workers=12)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_part, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_part, transform=train_transforms)#
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                    collate_fn=o_list_data_collate)
        return train_loader, train_sampler
    
    
    if args.phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms
    
    
    if args.phase == 'test':
        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return test_loader, val_transforms