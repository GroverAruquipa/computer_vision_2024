class Material:
    def __init__(self, name: str, width: float, length: float, folder_location: str):
        self.name = name
        self.width = width
        self.length = length
        self.folder_location = folder_location

    def compare_dimension(self, bb_width: float, bb_length: float) -> bool:
        ratio = 0.25
        # print(f"Largeur : {bb_width} Longueur : {bb_length})")
        min_width = self.width * (1 - ratio)
        max_width = self.width * (1 + ratio)
        min_length = self.length * (1 - ratio)
        max_length = self.length * (1 + ratio)

        # print(f"Largeur Min : {min_width} Largeur Max : {max_width})")
        # print(f"Longueur Min : {min_length} Longueur Max : {max_length})")

        width_match = (bb_width >= min_width) and (bb_width <= max_width)
        length_match = (bb_length >= min_length) and (bb_length <= max_length)

        # print(f"Witdh : {width_match} ")
        # print(f"Lenght : {length_match} ")

        return width_match and length_match
