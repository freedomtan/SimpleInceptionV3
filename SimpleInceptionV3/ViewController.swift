//
//  ViewController.swift
//  SimpleInceptionV3
//
//  Created by Koan-Sin Tan on 6/11/17.
//  Copyright © 2017. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML
import Vision

// Core ML MobileNet models could be converted from Keras models using script at https://github.com/freedomtan/coreml-mobilenet-models/.
// E.g., to get MobileNet 0.5/160,
//   > python mobilenets.py --alpha 0.50 --image_size 160

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, UITableViewDelegate, UITableViewDataSource, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var messageLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    @IBOutlet weak var tableView: UITableView!
    
    var session: AVCaptureSession = AVCaptureSession()
    var inputDevice: AVCaptureDevice!
    var deviceInput: AVCaptureDeviceInput!
    var previewLayer: AVCaptureVideoPreviewLayer!

    var model : VNCoreMLModel!
    var request: VNCoreMLRequest!
    var numberOfResults: Int = 0
    var results: [VNClassificationObservation] = []
    var startTimes: [TimeInterval] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.

        model = try! VNCoreMLModel(for: MobileNet_050_160().model)
        request = VNCoreMLRequest(model: model) { [weak self] request, error in
            let stopTicks = Date().timeIntervalSince1970
            let startTicks = self!.startTimes.remove(at: 0)

            guard let results = request.results as? [VNClassificationObservation],
                let topResult = results.first else {
                    fatalError("unexpected result type from VNCoreMLRequest")
            }
            self?.results = results
            self?.numberOfResults = results.count

            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.messageLabel.text = "\(Int(topResult.confidence * 100))% it's  \(topResult.identifier)"
                self?.fpsLabel.text = "\(round((1/(stopTicks - startTicks))*100)/100) fps"
                self?.tableView.reloadData()
            }
        }

        self.tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        self.setupCamera()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        messageLabel.text = "Analyzing Image…"
        guard let uiImage = info[UIImagePickerControllerOriginalImage] as? UIImage
            else { fatalError("no image from image picker") }
        
        // Show the image in the UI.
        imageView.image = uiImage
        
        guard let ciImage = CIImage(image: uiImage) else {
            fatalError("couldn't convert UIImage to CIImage")
        }
        
        labelImage(image: ciImage)
    }
    
    func labelImage(image: CIImage) {
        startTimes.append(Date().timeIntervalSince1970)
        // Run the Core ML MobileNet V1 classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).sync {
            do {
                try handler.perform([self.request])
            } catch {
                print(error)
            }
        }
    }
    
    @available(iOS 2.0, *)
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return (numberOfResults > 5) ? 5 : numberOfResults
    }
    
    @available(iOS 2.0, *)
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell:UITableViewCell = self.tableView.dequeueReusableCell(withIdentifier: "cell")!
        let string: NSMutableString = ""
        
        string.append("\(self.results[indexPath.row].confidence * 100)%: \(self.results[indexPath.row].identifier)")
        cell.textLabel?.text = string as String?
        
        return cell
    }
    
    @IBAction func takePicture(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        present(picker, animated: true)
    }
    
    @IBAction func chooseImage(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .savedPhotosAlbum
        present(picker, animated: true)
    }
    
    func setupCamera() {
        session.sessionPreset = AVCaptureSession.Preset.photo
        inputDevice = AVCaptureDevice.default(for: AVMediaType.video)!
        
        do {
            try deviceInput = AVCaptureDeviceInput(device: inputDevice)
            
            if (session.canAddInput(deviceInput)) {
                session.addInput(deviceInput)
                
                previewLayer = AVCaptureVideoPreviewLayer.init(session: session)
                previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
                
                let rootLayer: CALayer = self.view.layer
                rootLayer.masksToBounds = true
                previewLayer.frame = self.view.frame
                rootLayer.insertSublayer(previewLayer, at: 0)
                
                let videoDataOutput: AVCaptureVideoDataOutput = AVCaptureVideoDataOutput()
                videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCMPixelFormat_32BGRA]
                videoDataOutput.alwaysDiscardsLateVideoFrames = true
                videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "VdieoDataOutputQueue"))
                
                if (session.canAddOutput(videoDataOutput)) {
                    session.addOutput(videoDataOutput)
                    videoDataOutput.connection(with: AVMediaType.video)?.isEnabled = true
                    session.startRunning()
                } else {
                    print("Can't add videoDataOutput")
                }
            } else {
                print("Can't add deviceINput")
            }
        } catch let error as NSError {
            print("Error: \(error.domain)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let cvImage: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        let ciImage: CIImage = CIImage.init(cvImageBuffer: cvImage)
        self.labelImage(image: ciImage)
    }
}

