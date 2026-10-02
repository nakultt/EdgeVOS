package com.edgevos.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.edgevos.core.Mask

/**
 * Draws the tracker mask over the camera preview: green on (re-)grounded frames, red on tracked
 * frames, nothing while the object is absent. The mask is in analysis-frame coordinates (a 16:9
 * portrait frame); it is mapped with the same centre-crop scaling as PreviewView's FILL_CENTER.
 */
class OverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var bitmap: Bitmap? = null
    private var pixels = IntArray(0)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    private var visible = false

    /** Called on the UI thread. */
    fun show(mask: Mask?, grounded: Boolean) {
        if (mask == null) {
            visible = false
            invalidate()
            return
        }
        val bmp = bitmap?.takeIf { it.width == mask.w && it.height == mask.h }
            ?: Bitmap.createBitmap(mask.w, mask.h, Bitmap.Config.ARGB_8888).also { bitmap = it; pixels = IntArray(mask.w * mask.h) }
        val color = if (grounded) GREEN else RED
        for (i in pixels.indices) pixels[i] = if (mask.data[i]) color else 0
        bmp.setPixels(pixels, 0, mask.w, 0, 0, mask.w, mask.h)
        visible = true
        invalidate()
    }

    fun clear() = show(null, false)

    override fun onDraw(canvas: Canvas) {
        val bmp = bitmap ?: return
        if (!visible) return
        val scale = maxOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
        matrix.setScale(scale, scale)
        matrix.postTranslate((width - bmp.width * scale) / 2f, (height - bmp.height * scale) / 2f)
        canvas.drawBitmap(bmp, matrix, paint)
    }

    companion object {
        private const val GREEN = 0x8000E676.toInt()
        private const val RED = 0x80FF1744.toInt()
    }
}
