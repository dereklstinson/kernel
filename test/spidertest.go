package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"strings"

	"../../kernel"
	"golang.org/x/image/bmp"
)

func main() {
	stride := []int{2, 2}
	dilation := []int{1, 1}
	padding := []int{2, 2}
	firstkernel := [][]float64{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1},
	}
	convolution("spideronclover256.jpg", "spideredges22d11p11.bmp", firstkernel, stride, dilation, padding, false)
	//	convertjpgtobmp("spideronclover256.jpg")
	convolution("spideredges22d11p11.bmp", "reversespiders22d11p11.bmp", firstkernel, stride, dilation, padding, true)
}

func convolution(imagelocation string, newname string, k [][]float64, s, d, p []int, reverse bool) {
	file, err := os.Open(imagelocation)
	defer file.Close()
	if err != nil {
		fmt.Println("ON file")
		panic(err)
	}
	var img image.Image
	if strings.Contains(imagelocation, "bmp") {
		img, err = bmp.Decode(file)
		if err != nil {
			fmt.Println("on bmp encode")
			panic(err)
		}
	} else if strings.Contains(imagelocation, "jpg") {
		img, err = jpeg.Decode(file)
		if err != nil {
			fmt.Println("on jpeg encode")
			panic(err)
		}
	} else if strings.Contains(imagelocation, "png") {
		img, err = png.Decode(file)
		if err != nil {
			fmt.Println("on pgn encode")
			panic(err)
		}
	} else {
		panic("not supported file extention for decode. Must be bmp,jpg,or png")
	}
	var convimg image.Image
	if reverse {
		convimg = kernel.InverseConvolution(img, k, s, d, p)

	} else {
		convimg = kernel.Convolution(img, k, s, d, p)

	}

	newfile, err := os.Create(newname)
	defer newfile.Close()
	if err != nil {
		fmt.Println("ON newfile")
		panic(err)
	}
	if strings.Contains(newname, "bmp") {
		err = bmp.Encode(newfile, convimg)
		if err != nil {
			fmt.Println("on bmp.Encode")
			panic(err)
		}
	} else if strings.Contains(newname, "jpg") {
		err = jpeg.Encode(newfile, convimg, nil)
		if err != nil {
			fmt.Println("on jpeg.Encode")
			panic(err)
		}
	} else if strings.Contains(newname, "png") {
		err = png.Encode(newfile, convimg)
		if err != nil {
			fmt.Println("on png.Encode")
			panic(err)
		}
	} else {
		panic("not supported file extention for encode. Must be bmp,jpg,or png")
	}

}
func convertjpgtobmp(imagelocation string) {
	file, err := os.Open(imagelocation)
	defer file.Close()
	if err != nil {
		fmt.Println("ON file")
		panic(err)
	}
	img, err := jpeg.Decode(file)
	if err != nil {
		fmt.Println("on jpeg decode")
		panic(err)
	}
	newname := strings.TrimSuffix(imagelocation, "jpg") + "bmp"
	newfile, err := os.Create(newname)
	defer newfile.Close()
	if err != nil {
		fmt.Println("ON newfile")
		panic(err)
	}
	err = bmp.Encode(newfile, img)
	if err != nil {
		fmt.Println("on bmp.Encode")
		panic(err)
	}
}
