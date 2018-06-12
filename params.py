class PhotoTypes:
    FLOWER = 0
    PLANE = 1
    CAR = 2
    BIRD = 3
    CAT = 4
    DEER = 5
    DOG = 6
    FROG = 7
    HORSE = 8
    SHIP = 9
    TRUCK = 10


class Params(object):

    @staticmethod
    def update_params(cml_args):
        Params.ref_image_path = cml_args.ref_image_path
        Params.save_path = cml_args.save_path
        Params.number_mosaic_parts_horizontal = cml_args.number_parts_horizontal
        Params.arranging_way = cml_args.arranging_way
        Params.rand_criterion = cml_args.random_criterion
        Params.crossings_no = cml_args.crossings_no
        Params.identical_matching_pieces = cml_args.identical_matching_pieces
        Params.category = cml_args.category
