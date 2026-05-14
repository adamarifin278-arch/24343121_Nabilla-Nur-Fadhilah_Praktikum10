import cv2
import numpy as np
import matplotlib.pyplot as plt


def skeletonize(image):
    """Implementasi skeletonisasi tanpa ximgproc (pengganti thinning)"""
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = image.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def latihan_2():
    # Simulasikan citra dokumen dengan berbagai masalah
    doc = np.ones((200, 400), dtype=np.uint8) * 200  # Background terang

    # Tambahkan teks dengan berbagai masalah
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Teks normal
    cv2.putText(doc, 'Normal Text', (30, 50), font, 0.7, 50, 2)

    # Teks dengan noise (garis putus-putus)
    for i in range(0, 100, 5):
        cv2.line(doc, (30 + i, 80), (30 + i, 85), 50, 1)

    # Teks dengan lubang (karakter patah)
    cv2.putText(doc, 'Broken Text', (30, 120), font, 0.7, 50, 2)
    # Buat lubang di tengah karakter
    cv2.rectangle(doc, (80, 110), (90, 115), 200, -1)

    # Tambahkan background noise
    noise = np.random.normal(0, 30, doc.shape)
    doc = np.clip(doc.astype(float) + noise, 0, 255).astype(np.uint8)

    # Preprocessing pipeline untuk OCR
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Step 1: Original
    axes[0, 0].imshow(doc, cmap='gray')
    axes[0, 0].set_title('Original Document\n(with various defects)')
    axes[0, 0].axis('off')

    # Step 2: Binarization
    _, binary = cv2.threshold(doc, 150, 255, cv2.THRESH_BINARY_INV)
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Step 1: Binarization')
    axes[0, 1].axis('off')

    # Step 3: Noise removal dengan opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    axes[0, 2].imshow(cleaned, cmap='gray')
    axes[0, 2].set_title('Step 2: Noise Removal\n(Opening 1x1, 2 iterations)')
    axes[0, 2].axis('off')

    # Step 4: Connect broken characters dengan closing
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_horizontal)
    axes[1, 0].imshow(connected, cmap='gray')
    axes[1, 0].set_title('Step 3: Connect Characters\n(Closing 3x1 horizontal)')
    axes[1, 0].axis('off')

    # Step 5: Enhance stroke thickness dengan dilation
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    enhanced = cv2.dilate(connected, kernel_vertical, iterations=1)
    axes[1, 1].imshow(enhanced, cmap='gray')
    axes[1, 1].set_title('Step 4: Stroke Enhancement\n(Dilation 1x2 vertical)')
    axes[1, 1].axis('off')

    # Step 6: Final result
    final_result = enhanced
    axes[1, 2].imshow(final_result, cmap='gray')
    axes[1, 2].set_title('Step 5: Final Result\n(Ready for OCR)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('latihan_2_output.png', dpi=150, bbox_inches='tight')
    print("Gambar disimpan ke: latihan_2_output.png (folder yang sama dengan script)")
    plt.show()

    # Quantitative evaluation
    def calculate_text_quality(original_binary, processed_binary):
        """Calculate simple text quality metrics"""

        def count_connected_components(image):
            num_labels, _ = cv2.connectedComponents(image)
            return num_labels - 1  # Subtract background

        orig_components = count_connected_components(original_binary)
        proc_components = count_connected_components(processed_binary)

        def average_stroke_thickness(image):
            """Hitung rata-rata ketebalan stroke menggunakan skeletonisasi manual"""
            skeleton = skeletonize(image)
            stroke_pixels = np.sum(image == 255)
            skeleton_pixels = np.sum(skeleton == 255)
            return stroke_pixels / skeleton_pixels if skeleton_pixels > 0 else 0

        orig_thickness = average_stroke_thickness(original_binary)
        proc_thickness = average_stroke_thickness(processed_binary)

        improvement_cc = (
            (orig_components - proc_components) / orig_components * 100
            if orig_components > 0 else 0
        )
        improvement_st = (
            (proc_thickness - orig_thickness) / orig_thickness * 100
            if orig_thickness > 0 else 0
        )

        return {
            'connected_components': {
                'original': orig_components,
                'processed': proc_components,
                'improvement': improvement_cc
            },
            'stroke_thickness': {
                'original': orig_thickness,
                'processed': proc_thickness,
                'improvement': improvement_st
            }
        }

    quality_metrics = calculate_text_quality(binary, final_result)

    print("\nOCR PREPROCESSING QUALITY METRICS:")
    print("=" * 50)
    print("Connected Components Analysis:")


    
    print(f"  Original : {quality_metrics['connected_components']['original']} components")
    print(f"  Processed: {quality_metrics['connected_components']['processed']} components")
    print(f"  Improvement: {quality_metrics['connected_components']['improvement']:.1f}% reduction")

    print("\nStroke Thickness Analysis:")
    print(f"  Original : {quality_metrics['stroke_thickness']['original']:.2f} (avg pixels)")
    print(f"  Processed: {quality_metrics['stroke_thickness']['processed']:.2f} (avg pixels)")
    print(f"  Improvement: {quality_metrics['stroke_thickness']['improvement']:.1f}% increase")

    print("\nCONCLUSION:")
    print("Operasi morfologi dapat secara signifikan meningkatkan")
    print("kualitas citra teks untuk OCR dengan:")
    print("1. Menghilangkan noise")
    print("2. Menyambungkan karakter yang patah")
    print("3. Mempertebal stroke yang tipis")


# Jalankan latihan 2
latihan_2()