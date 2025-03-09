import pygame

colors = {
    0: (204, 192, 179), 
    2: (236, 224, 223),
    4: (237, 232, 211),
    8: (238, 190, 136),
    16: (237, 169, 115),
    32: (235, 152, 112),
    64: (232, 137, 78),
    128: (234, 209, 132),
    256: (235, 209, 132),
    512: (198, 167, 88),
    1024: (191, 163, 73),
    2048: (217, 181, 71),
    4096: (229, 136, 119),
    8192: (227, 116, 104),
    16384: (218, 104, 70),
    32768: (135, 183, 217),
    65536: (117, 167, 223),
    131072: (51, 105, 161)
}   

class Tile:   
    def __init__(self,value, font):
        self.value = value
        self.color = colors.get(value)
        self.font = font
        
    def draw(self, screen, x, y, tile_size):
        if self.value != 0:
            text = self.font.render(str(self.value), True, (0, 0, 0))
        else:
            text = self.font.render('', True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + tile_size // 2, y + tile_size // 2))
        pygame.draw.rect(screen, self.color, pygame.Rect(x, y, tile_size, tile_size))
        screen.blit(text, text_rect)
    
    def get_tile_value(self):
        return self.value
    
    def set_tile_value(self, value):
        self.value = value
        self.color = colors[value]