package kernel

import (
	"image"
	"image/color"
	"sync"
)

//Convolution does the convolution operation per channel on an image. threads will parallelize the convolution
func Convolution(img image.Image, kernel [][]float64, stride, dilation, padding []int, zeronegatives bool, threads bool) image.Image {
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
	arrayout := create3darray(op[0], op[1], 3)
	if threads {
		var wg sync.WaitGroup
		for oh, xhps := 0, -padding[0]; oh < op[0]; oh, xhps = oh+1, xhps+stride[0] {

			for ow, xwps := 0, -padding[1]; ow < op[1]; ow, xwps = ow+1, xwps+stride[1] {

				d1 := dilation[1]
				d0 := dilation[0]
				wg.Add(1)
				go func(oh, ow, xhps, xwps, d0, d1 int) {
					var (
						sr float64
						sg float64
						sb float64
					)
					khlast := len(kernel) - 1
					for kh := 0; kh < len(kernel); kh++ {
						xhpsd := kh*d0 + xhps
						kwlast := len(kernel[kh]) - 1
						for kw := 0; kw < len(kernel[kh]); kw++ {
							xwpsd := kw*d1 + xwps
							if xwpsd >= 0 && xwpsd < x[1] && xhpsd >= 0 && xhpsd < x[0] {
								r, g, b, _ := img.At(xwpsd, xhpsd).RGBA()
								kval := kernel[khlast-kh][kwlast-kw]
								sr += (float64)(r) * kval
								sg += (float64)(g) * kval
								sb += (float64)(b) * kval
							}

						}

					}

					arrayout[oh][ow][0] = sr
					arrayout[oh][ow][1] = sg
					arrayout[oh][ow][2] = sb
					wg.Done()
				}(oh, ow, xhps, xwps, d0, d1)

			}

		}
		wg.Wait()
	} else {
		for oh, xhps := 0, -padding[0]; oh < op[0]; oh, xhps = oh+1, xhps+stride[0] {

			for ow, xwps := 0, -padding[1]; ow < op[1]; ow, xwps = ow+1, xwps+stride[1] {

				d1 := dilation[1]
				d0 := dilation[0]

				var (
					sr float64
					sg float64
					sb float64
				)
				khlast := len(kernel) - 1
				for kh := 0; kh < len(kernel); kh++ {
					xhpsd := kh*d0 + xhps
					kwlast := len(kernel[kh]) - 1
					for kw := 0; kw < len(kernel[kh]); kw++ {
						xwpsd := kw*d1 + xwps
						if xwpsd >= 0 && xwpsd < x[1] && xhpsd >= 0 && xhpsd < x[0] {
							r, g, b, _ := img.At(xwpsd, xhpsd).RGBA()
							kval := kernel[khlast-kh][kwlast-kw]
							sr += (float64)(r) * kval
							sg += (float64)(g) * kval
							sb += (float64)(b) * kval
						}

					}

				}

				arrayout[oh][ow][0] = sr
				arrayout[oh][ow][1] = sg
				arrayout[oh][ow][2] = sb

			}

		}
	}
	newimage := array3dtoimg(arrayout, zeronegatives)
	arrayout = nil
	return newimage
}

//InverseConvolution does reverse convolution operation per channel on an image.
//This is kind of like the back propagation of an CNN.
func InverseConvolution(img image.Image, kernel [][]float64, stride, dilation, padding []int, zeronegatives bool) image.Image {
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
			khlast := len(kernel) - 1
			for kh := 0; kh < len(kernel); kh++ {
				xhpsd := kh*dilation[0] + xhps
				kwlast := len(kernel[kh]) - 1
				for kw := 0; kw < len(kernel[kh]); kw++ {
					xwpsd := kw*dilation[1] + xwps
					if xwpsd >= 0 && xwpsd < ox[1] && xhpsd >= 0 && xhpsd < ox[0] {
						kval := kernel[khlast-kh][kwlast-kw]
						array3d[xhpsd][xwpsd][0] += dr * kval
						array3d[xhpsd][xwpsd][1] += dg * kval
						array3d[xhpsd][xwpsd][2] += db * kval

					}

				}

			}

		}

	}
	return array3dtoimg(array3d, zeronegatives)

}

//InverseCCorelation does the reverse cross corelation operation per channel on an image.
//This is kind of like the back propagation of an CNN.
func InverseCCorelation(img image.Image, kernel [][]float64, stride, dilation, padding []int, zeronegatives bool) image.Image {
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
	return array3dtoimg(array3d, zeronegatives)

}

/*
func NormalizeImage(img image.Image) image.Image {
	imgarray:=imgto3darray(img)
	max:=-9999999.0
	min:=9999999.0
	avg:=0.0
	counter:=0 //im lazy
	for i:=range imgarray{
		for j:=range imgarray[0]{
			for k:=range imgarray[0][0]{
val:=imgarray[i][j][k]
if min>val{
	min=val
}
if max<val{
	max=val
}
avg+=val
counter++

			}
		}
	}

}
*/

//CrossCorelation does the convolution operation per channel on an image. If the kernel is small and picture is small then don't bother setting the threads to big.
func CrossCorelation(img image.Image, kernel [][]float64, stride, dilation, padding []int, zeronegatives bool, threads bool) image.Image {
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

	arrayout := create3darray(op[0], op[1], 3)
	if threads {

		var wg sync.WaitGroup
		for oh, xhps := 0, -padding[0]; oh < op[0]; oh, xhps = oh+1, xhps+stride[0] {

			for ow, xwps := 0, -padding[1]; ow < op[1]; ow, xwps = ow+1, xwps+stride[1] {

				d1 := dilation[1]
				d0 := dilation[0]
				wg.Add(1)
				go func(oh, ow, xhps, xwps, d0, d1 int) {
					var (
						sr float64
						sg float64
						sb float64
					)
					for kh := 0; kh < len(kernel); kh++ {
						xhpsd := kh*d0 + xhps
						for kw := 0; kw < len(kernel[kh]); kw++ {
							xwpsd := kw*d1 + xwps
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
					wg.Done()
				}(oh, ow, xhps, xwps, d0, d1)

			}

		}
		wg.Wait()
	} else {

		for oh, xhps := 0, -padding[0]; oh < op[0]; oh, xhps = oh+1, xhps+stride[0] {

			for ow, xwps := 0, -padding[1]; ow < op[1]; ow, xwps = ow+1, xwps+stride[1] {

				d1 := dilation[1]
				d0 := dilation[0]

				var (
					sr float64
					sg float64
					sb float64
				)
				for kh := 0; kh < len(kernel); kh++ {
					xhpsd := kh*d0 + xhps
					for kw := 0; kw < len(kernel[kh]); kw++ {
						xwpsd := kw*d1 + xwps
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

			}

		}
	}
	newimage := array3dtoimg(arrayout, zeronegatives)
	arrayout = nil
	return newimage
}

func array3dtoimg(input [][][]float64, zeronegatives bool) image.Image {
	y := len(input)
	x := len(input[0])
	c := len(input[0][0])
	var (
		min = 99999999.0
		max = -99999999.0
	)
	if zeronegatives {
		for i := 0; i < y; i++ {
			for j := 0; j < x; j++ {
				for k := 0; k < c; k++ {

					if input[i][j][k] < 0 {
						input[i][j][k] = 0
					}

					if min > input[i][j][k] {
						min = input[i][j][k]
					}
					if max < input[i][j][k] {
						max = input[i][j][k]
					}
				}
			}
		}
	} else {
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
	if zeronegatives {
		if max > 255 {
			for i := 0; i < y; i++ {
				for j := 0; j < x; j++ {
					for k := 0; k < c; k++ {
						if input[i][j][k] > 255 {
							input[i][j][k] = 255
						}
						//	input[i][j][k] = (input[i][j][k] * 255) / max

					}
				}
			}
		}
	} else {
		if max > 255 {
			for i := 0; i < y; i++ {
				for j := 0; j < x; j++ {
					for k := 0; k < c; k++ {

						input[i][j][k] = (input[i][j][k] * 255) / max

					}
				}
			}
		}
	}

	switch len(input[0][0]) {
	case 1:
		imgout := image.NewRGBA(image.Rect(0, 0, x, y))
		for i := 0; i < y; i++ {
			for j := 0; j < x; j++ {
				usr := (uint8)(input[i][j][0])
				usg := (uint8)(input[i][j][0])
				usb := (uint8)(input[i][j][0])
				imgout.Set(j, i, color.RGBA{usr, usg, usb, 255})
			}
		}
		return imgout
	case 2:
		imgout := image.NewRGBA(image.Rect(0, 0, x, y))
		for i := 0; i < y; i++ {
			for j := 0; j < x; j++ {
				usr := (uint8)(input[i][j][0])
				usg := (uint8)(input[i][j][1])
				usb := (uint8)((input[i][j][0] + input[i][j][1]) / 2)
				imgout.Set(j, i, color.RGBA{usr, usg, usb, 255})
			}
		}
		return imgout
	case 3:
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
	case 4:
		imgout := image.NewRGBA(image.Rect(0, 0, x, y))
		for i := 0; i < y; i++ {
			for j := 0; j < x; j++ {
				usr := (uint8)(input[i][j][0])
				usg := (uint8)(input[i][j][1])
				usb := (uint8)(input[i][j][2])
				usa := (uint8)(input[i][j][3])
				imgout.Set(j, i, color.RGBA{usr, usg, usb, usa})
			}
		}
		return imgout

	}
	return nil
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
func imgto3darray(img image.Image) [][][]float64 {
	w := img.Bounds().Max.X
	h := img.Bounds().Max.Y
	array := create3darray(h, w, 3)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			r, g, b, _ := img.At(w, h).RGBA()
			array[i][j][0] = float64(r)
			array[i][j][1] = float64(g)
			array[i][j][2] = float64(b)
		}
	}
	return array

}
func output(x, w, s, d, p int) (y int) {
	return ((x + 2*p - ((w-1)*d + 1)) / s) + 1

}
func reverseoutput(x, w, s, d, p int) (y int) {
	return (x-1)*s - 2*p + ((w-1)*d + 1)
}
