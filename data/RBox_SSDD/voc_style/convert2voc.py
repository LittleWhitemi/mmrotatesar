import os
path = 'test_offshore_labels.csv'
path_save = 'Annotations_test_offshore/'
with open(path, 'r') as csv_file:
    contents = csv_file.read()
    print('contents = ', contents)
    image_list = contents.split('\n\n')[:-1]
    # print('len(image_list) = ', len(image_list))
    for one_image in image_list:
        lines = one_image.split('\n')

        image_infor = lines[0]
        ships = lines[1:]

        image_name = image_infor.split(',')[0]
        image_width = image_infor.split(',')[-1].split('x')[0]
        image_height = image_infor.split(',')[-1].split('x')[1]

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        with open(path_save + image_name.replace('jpg', 'xml'), 'w') as xml_file:
            xml_file.write('<annotation>\n')
            xml_file.write('\t<folder>{}</folder>\n'.format('JPEGImages'))
            xml_file.write('\t<filename>{}</filename>\n'.format(image_name))
            xml_file.write('\t<path>{}</path>\n'.format('JPEGImages'))
            xml_file.write('\t<size>\n')
            xml_file.write('\t\t<width>{}</width>\n'.format(image_width.replace(' ', '')))
            xml_file.write('\t\t<height>{}</height>\n'.format(image_height.replace(' ', '')))
            xml_file.write('\t\t<depth>{}</depth>\n'.format('3'))
            xml_file.write('\t</size>\n')

            for one_object in ships:
                xml_file.write('\t<object>\n')
                bbox_xmin = one_object.split(',')[2]
                bbox_xmax = one_object.split(',')[4]
                bbox_ymin = one_object.split(',')[6]
                bbox_ymax = one_object.split(',')[8]
                bbox_w = one_object.split(',')[10]
                bbox_h = one_object.split(',')[12]
                rotated_bbox_cx = one_object.split(',')[14]
                rotated_bbox_cy = one_object.split(',')[16]
                rotated_bbox_w = one_object.split(',')[18]
                rotated_bbox_h = one_object.split(',')[20]
                rotated_bbox_theta = one_object.split(',')[22]
                x1 = one_object.split(',')[24]
                y1 = one_object.split(',')[26]
                x2 = one_object.split(',')[28]
                y2 = one_object.split(',')[30]
                x3 = one_object.split(',')[32]
                y3 = one_object.split(',')[34]
                x4 = one_object.split(',')[36]
                y4 = one_object.split(',')[38]



                xml_file.write('\t\t<name>{}</name>\n'.format('ship'))
                xml_file.write('\t\t<pose>{}</pose>\n'.format('Unspecified'))
                xml_file.write('\t\t<truncated>{}</truncated>\n'.format('0'))
                xml_file.write('\t\t<difficult>{}</difficult>\n'.format('0'))

                # xml_file.write('\t\t<bndbox>\n')
                # xml_file.write('\t\t\t<xmin>{}</xmin>\n'.format(bbox_xmin.replace(' ', '')))
                # xml_file.write('\t\t\t<ymin>{}</ymin>\n'.format(bbox_ymin.replace(' ', '')))
                # xml_file.write('\t\t\t<xmax>{}</xmax>\n'.format(bbox_xmax.replace(' ', '')))
                # xml_file.write('\t\t\t<ymax>{}</ymax>\n'.format(bbox_ymax.replace(' ', '')))
                # xml_file.write('\t\t\t<bbox_w>{}</bbox_w>\n'.format(bbox_w.replace(' ', '')))
                # xml_file.write('\t\t\t<bbox_h>{}</bbox_h>\n'.format(bbox_h.replace(' ', '')))
                # xml_file.write('\t\t</bndbox>\n')

                xml_file.write('\t\t<rotated_bndbox>\n')
                xml_file.write('\t\t\t<rotated_bbox_cx>{}</rotated_bbox_cx>\n'.format(rotated_bbox_cx.replace(' ', '')))
                xml_file.write('\t\t\t<rotated_bbox_cy>{}</rotated_bbox_cy>\n'.format(rotated_bbox_cy.replace(' ', '')))
                xml_file.write('\t\t\t<rotated_bbox_w>{}</rotated_bbox_w>\n'.format(rotated_bbox_w.replace(' ', '')))
                xml_file.write('\t\t\t<rotated_bbox_h>{}</rotated_bbox_h>\n'.format(rotated_bbox_h.replace(' ', '')))
                xml_file.write('\t\t\t<rotated_bbox_theta>{}</rotated_bbox_theta>\n'.format(rotated_bbox_theta.replace(' ', '')))
                xml_file.write('\t\t\t<x1>{}</x1>\n'.format(x1.replace(' ', '')))
                xml_file.write('\t\t\t<y1>{}</y1>\n'.format(y1.replace(' ', '')))
                xml_file.write('\t\t\t<x2>{}</x2>\n'.format(x2.replace(' ', '')))
                xml_file.write('\t\t\t<y2>{}</y2>\n'.format(y2.replace(' ', '')))
                xml_file.write('\t\t\t<x3>{}</x3>\n'.format(x3.replace(' ', '')))
                xml_file.write('\t\t\t<y3>{}</y3>\n'.format(y3.replace(' ', '')))
                xml_file.write('\t\t\t<x4>{}</x4>\n'.format(x4.replace(' ', '')))
                xml_file.write('\t\t\t<y4>{}</y4>\n'.format(y4.replace(' ', '')))
                xml_file.write('\t\t</rotated_bndbox>\n')

                # xml_file.write('\t\t<segm>\n')
                # segm_list = one_object.split('segm,')[-1].split('),')[:-1]

                # for i, one_segm_point in enumerate(segm_list):
                    # xml_file.write('\t\t\t<point-{}>{})</point-{}>\n'.format(i + 1, one_segm_point.replace(' ', ''), i + 1).replace('(', '').replace(')', ''))
                # xml_file.write('\t\t</segm>\n')
                xml_file.write('\t</object>\n')
            xml_file.write('</annotation>')

        print('image_infor = ', image_infor)
        print('ships = ', ships)
