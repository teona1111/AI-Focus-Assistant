$fn = 64;

dolzina = 85;
sirina = 58;
visina = 22;

debelina_stene = 2;

usb_sirina = 14;
usb_visina = 8;

krog_r = 3;

sponka_dolzina = 55;
sponka_sirina = 24;
sponka_visina = 12;
ukrivljenost = 6;

lok_sirina = 14;
lok_visina = 8;


module ukrivljen_blok(dolz, sir, vis, padec) {
    koraki = 8;

    for (i = [0 : koraki - 1]) {
        x1 = -dolz / 2 + i * dolz / koraki;
        x2 = -dolz / 2 + (i + 1) * dolz / koraki;

        z1 = -padec * pow(x1 / (dolz / 2), 2);
        z2 = -padec * pow(x2 / (dolz / 2), 2);

        hull() {
            translate([x1, 0, z1])
                cube([dolz / koraki, sir, vis], center = true);

            translate([x2, 0, z2])
                cube([dolz / koraki, sir, vis], center = true);
        }
    }
}


difference() {

    union() {

        cube([dolzina, sirina, visina], center = true);

        translate([0, 0, -visina / 2 - 5])
            difference() {

                ukrivljen_blok(
                    sponka_dolzina,
                    sponka_sirina,
                    sponka_visina,
                    ukrivljenost
                );

                ukrivljen_blok(
                    sponka_dolzina + 4,
                    lok_sirina,
                    lok_visina,
                    ukrivljenost
                );

                translate([0, 0, -7])
                    ukrivljen_blok(
                        sponka_dolzina + 4,
                        lok_sirina + 5,
                        8,
                        ukrivljenost
                    );
            }
    }

    translate([0, 0, debelina_stene])
        cube([
            dolzina - 2 * debelina_stene,
            sirina - 2 * debelina_stene,
            visina
        ], center = true);

    translate([-dolzina / 2, 0, 2])
        cube([8, usb_sirina, usb_visina], center = true);

    translate([dolzina / 2, 0, 2])
        cube([8, usb_sirina, usb_visina], center = true);

    translate([dolzina / 2, -14, 7])
        rotate([0, 90, 0])
            cylinder(h = 8, r = krog_r, center = true);

    for (x = [-25 : 10 : 25]) {
        translate([x, 0, visina / 2])
            cylinder(h = 6, r = 2, center = true);
    }

    for (x = [-32, 32]) {
        for (y = [-20, 20]) {
            translate([x, y, visina / 2])
                cylinder(h = 6, r = 1.6, center = true);
        }
    }
}