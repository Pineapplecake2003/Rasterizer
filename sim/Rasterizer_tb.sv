`timescale 1ns/10ps

module Rasterizer_tb (
);
    logic clk;
    logic rst;
    typedef logic [31:0] int32_t;
    typedef logic [31:0] fp32_t;

    typedef struct packed {
        int32_t x;
        int32_t y;
        fp32_t z_inv;
        fp32_t brightness;
    } point_t;

    point_t p[0:2];
    int fd;
    int r;
    int i;
    string golden_path;
    initial begin
        if (!$value$plusargs("golden_path=%s", golden_path))
            golden_path = "../golden/points.txt";

        $display("Using golden_path = %s", golden_path);

        fd = $fopen(golden_path, "r");
        if (fd == 0)
        $fatal("Cannot open %s", golden_path);

        for (i = 0; i < 3; i++) begin
            r = $fscanf(fd, "%h\n", p[i].x);
            r = $fscanf(fd, "%h\n", p[i].y);
            r = $fscanf(fd, "%h\n", p[i].z_inv);
            r = $fscanf(fd, "%h\n", p[i].brightness);

            if (r != 1)
                $fatal("Read error at point %0d", i);
        end

        $display("Points loaded:");
        for (i = 0; i < 3; i++) begin
            $display("p[%0d] x=%d y=%d z_inv=%f brightness=%f",
                    i, $signed(p[i].x), $signed(p[i].y), $bitstoshortreal(p[i].z_inv), $bitstoshortreal(p[i].brightness));
        end

        $fclose(fd);
        $finish;
    end
endmodule