import cv2
import numpy as np
import argparse

def measure_object(image_path, known_width=None, ref_object_width_mm=None):
    """
    Measure the length and width of objects in an image.
    
    Parameters:
    image_path (str): Path to the input image
    known_width (float, optional): Known width of reference object in pixels
    ref_object_width_mm (float, optional): Real-world width of reference object in mm
    
    Returns:
    image: Original image with measurements drawn
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw on
    output = image.copy()
    
    # Reference object should be the first/leftmost object in the image
    # (assuming the reference object is placed to the left)
    pixels_per_mm = None
    
    if len(contours) > 0:
        # Sort contours by x-coordinate (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # If we're calibrating with a reference object
        if known_width is not None and ref_object_width_mm is not None:
            # The first contour should be our reference object
            ref_contour = contours[0]
            ref_box = cv2.minAreaRect(ref_contour)
            ref_box_points = cv2.boxPoints(ref_box)
            ref_box_points = np.int0(ref_box_points)
            
            # Calculate the pixels per millimeter using the reference object
            pixels_per_mm = known_width / ref_object_width_mm
            
            # Draw the reference object
            cv2.drawContours(output, [ref_box_points], 0, (0, 255, 0), 2)
            cv2.putText(output, "Reference Object", (int(ref_box[0][0] - 15), int(ref_box[0][1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Skip the reference object in the measurements
            contours = contours[1:]
        
        # Process each contour (object)
        for i, contour in enumerate(contours):
            # Skip small contours that might be noise
            if cv2.contourArea(contour) < 100:
                continue
            
            # Calculate the rotated rectangle (minimum area rectangle)
            rect = cv2.minAreaRect(contour)
            
            # Get the box points
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get width and height of the rectangle
            width = rect[1][0]
            height = rect[1][1]
            
            # Make sure width is always the longer dimension
            if width < height:
                width, height = height, width
                
            # Draw the rectangle
            cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
            
            # Display dimensions
            if pixels_per_mm is not None:
                # Convert measurements to mm
                width_mm = width / pixels_per_mm
                height_mm = height / pixels_per_mm
                
                # Display real-world measurements
                cv2.putText(output, f"Width: {width_mm:.1f}mm", 
                            (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output, f"Height: {height_mm:.1f}mm", 
                            (int(rect[0][0] - 15), int(rect[0][1] + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Display pixel measurements if no calibration
                cv2.putText(output, f"Width: {width:.1f}px", 
                            (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output, f"Height: {height:.1f}px", 
                            (int(rect[0][0] - 15), int(rect[0][1] + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Measure objects in an image using computer vision')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--known_width', type=float, help='Known width of reference object in pixels')
    parser.add_argument('--ref_width_mm', type=float, help='Real-world width of reference object in mm')
    parser.add_argument('--output', default='output.jpg', help='Output image path')
    
    args = parser.parse_args()
    
    # Process the image
    result = measure_object(args.image_path, args.known_width, args.ref_width_mm)
    
    if result is not None:
        # Show the result
        cv2.imshow('Object Measurement', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the output image
        cv2.imwrite(args.output, result)
        print(f"Results saved to {args.output}")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()