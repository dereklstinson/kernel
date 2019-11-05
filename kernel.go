package kernel

import (
	"fmt"
	"image"
	"image/color"
)

func InverseConvolution(img image.Image, kernel [][]float64, stride, dilation, padding []int) image.Image {
	if len(stride) != 2 || len(dilation) != 2 || len(padding) != 2 {
		return nil
	}
	ox := make([]int, 2)
	w := make([]int, 2)
	y := make([]int, 2)
	y[0] = img.Bounds().Max.Y
	y[1] = img.Bounds().Max.X
	w[0] = len(kernel)
	w[1] = len(kernel[0])
	for i := range ox {

		ox[i] = reverseoutput(y[i], w[i], stride[i], dilation[i], padding[i])
	}
	array3d := create3darray(ox[0], ox[1], 3)

	for oh, xhps := 0, -padding[0]; oh < y[0]; oh, xhps = oh+1, xhps+stride[0] {

		for ow, xwps := 0, -padding[1]; ow < y[1]; ow, xwps = ow+1, xwps+stride[1] {
			ddr, ddg, ddb, _ := img.At(ow, oh).RGBA()
			dr, dg, db := (float64)(ddr), (float64)(ddg), (float64)(ddb)
			for kh := 0; kh < len(kernel); kh++ {
				xhpsd := kh*dilation[0] + xhps
				for kw := 0; kw < len(kernel[kh]); kw++ {
					xwpsd := kw*dilation[1] + xwps
					if xwpsd >= 0 && xwpsd < ox[1] && xhpsd >= 0 && xhpsd < ox[0] {

						array3d[xhpsd][xwpsd][0] += dr * kernel[kh][kw]
						array3d[xhpsd][xwpsd][1] += dg * kernel[kh][kw]
						array3d[xhpsd][xwpsd][2] += db * kernel[kh][kw]

					}

				}

			}

		}

	}
	return array3dtoimg(array3d)

}
func floatarray3dtoimage(a [][][]float32, x, y int) image.Image {
	return nil
}
func Convolution(img image.Image, kernel [][]float64, stride, dilation, padding []int) image.Image {
	if len(stride) != 2 || len(dilation) != 2 || len(padding) != 2 {
		return nil
	}
	op := make([]int, 2)
	w := make([]int, 2)
	x := make([]int, 2)
	x[0] = img.Bounds().Max.Y
	x[1] = img.Bounds().Max.X
	w[0] = len(kernel)
	w[1] = len(kernel[0])
	for i := range op {

		op[i] = output(x[i], w[i], stride[i], dilation[i], padding[i])
	}
	fmt.Println("Output", op)
	arrayout := create3darray(op[0], op[1], 3)
	//imgout := image.NewRGBA(image.Rect(0, 0, op[1], op[0]))

	for oh, xhps := 0, -padding[0]; oh < op[0]; oh, xhps = oh+1, xhps+stride[0] {

		for ow, xwps := 0, -padding[1]; ow < op[1]; ow, xwps = ow+1, xwps+stride[1] {
			var (
				sr float64
				sg float64
				sb float64
			)
			for kh := 0; kh < len(kernel); kh++ {
				xhpsd := kh*dilation[0] + xhps
				for kw := 0; kw < len(kernel[kh]); kw++ {
					xwpsd := kw*dilation[1] + xwps
					if xwpsd >= 0 && xwpsd < x[1] && xhpsd >= 0 && xhpsd < x[0] {
						r, g, b, _ := img.At(xwpsd, xhpsd).RGBA()

						sr += (float64)(r) * kernel[kh][kw]
						sg += (float64)(g) * kernel[kh][kw]
						sb += (float64)(b) * kernel[kh][kw]
					}

				}

			}
			arrayout[oh][ow][0] = sr
			arrayout[oh][ow][1] = sg
			arrayout[oh][ow][2] = sb

			/*
				usr := (uint8)(sr / 257)
				usg := (uint8)(sg / 257)
				usb := (uint8)(sb / 257)
				imgout.Set(ow, oh, color.RGBA{usr, usg, usb, 255})
			*/
		}

	}
	return array3dtoimg(arrayout)
}

func array3dtoimg(input [][][]float64) image.Image {
	y := len(input)
	x := len(input[0])
	c := len(input[0][0])
	var (
		min = 99999999.0
		max = -99999999.0
	)
	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			for k := 0; k < c; k++ {
				if min > input[i][j][k] {
					min = input[i][j][k]
				}
				if max < input[i][j][k] {
					max = input[i][j][k]
				}
			}
		}
	}
	if min < 0 {
		max -= min
		for i := 0; i < y; i++ {
			for j := 0; j < x; j++ {
				for k := 0; k < c; k++ {

					input[i][j][k] -= min

				}
			}
		}
	}
	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			for k := 0; k < c; k++ {

				input[i][j][k] = (input[i][j][k] * 255) / max

			}
		}
	}

	imgout := image.NewRGBA(image.Rect(0, 0, x, y))
	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			usr := (uint8)(input[i][j][0])
			usg := (uint8)(input[i][j][1])
			usb := (uint8)(input[i][j][2])
			imgout.Set(j, i, color.RGBA{usr, usg, usb, 255})
		}
	}
	return imgout
}
func create3darray(y, x, c int) [][][]float64 {
	out := make([][][]float64, y)
	for i := range out {
		out[i] = make([][]float64, x)
		for j := range out[i] {
			out[i][j] = make([]float64, c)
		}
	}
	return out
}
func output(x, w, s, d, p int) (y int) {
	return ((x + 2*p - ((w-1)*d + 1)) / s) + 1

}
func reverseoutput(x, w, s, d, p int) (y int) {
	return (x-1)*s - 2*p + ((w-1)*d + 1)
}
