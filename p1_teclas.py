import pygame

pygame.init()
screen = pygame.display.set_mode((420, 80))
pygame.display.set_caption("haz clic aqui y pulsa WASD")
font = pygame.font.SysFont("consolas", 20)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit
    k = pygame.key.get_pressed()
    text = (
        f"W={int(k[pygame.K_w])} S={int(k[pygame.K_s])} "
        f"A={int(k[pygame.K_a])} D={int(k[pygame.K_d])}"
    )
    screen.fill((0, 0, 0))
    screen.blit(font.render(text, True, (220, 220, 220)), (12, 28))
    pygame.display.flip()