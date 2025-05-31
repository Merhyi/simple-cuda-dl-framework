
#include <iostream>
#include "./datasets/mnist_dataset.hh"
#include "nn_utils/bce_cost.hh"
#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/sigmoid_activation.hh"
#include "layers/leaky_relu_activation.hh"
#include "layers/relu_activation.hh"
#include "layers/tanh_activation.hh"

#include "optimizers/adam_optimizer.hh"


static void printMatrixCUDA(const Matrix& mat_to_print) {
    // ... rest of the logic using mat_to_print.data_device.get() and cudaMemcpy ...
    int count = mat_to_print.shape.x * mat_to_print.shape.y;

    std::vector<float> h_data(count);
    printf("Pointer: %p\n", mat_to_print.data_device.get());
    cudaError_t err = cudaMemcpy(h_data.data(), mat_to_print.data_device.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
    for (auto data : h_data)
    {
        std::cout << data << "\t";
    }
}





int main()
{
    const int batch_size = 64;
    const int image_size = 784;
    const int hidden_size = 256;
    const int latent_size = 64;
    const float learning_rate = 0.02;
    const int num_epochs = 300;
 
    MNISTDataset train_dataset(
        batch_size, // batch_size
        "data/mnist/train-images.idx3-ubyte",
        "data/mnist/train-labels.idx1-ubyte",
        3000, // load all training samples
        true,  // normalize
        false  // do not one-hot encode (use integer labels)
     );
    
    BCECost bce_cost;

    // architecture of discriminator
    NeuralNetwork discriminator;
    discriminator.addLayer(new LinearLayer("linear-1", Shape(image_size, hidden_size)));
    discriminator.addLayer(new LeakyReLUActivation("leaky-relu-1", 0.2));
    discriminator.addLayer(new LinearLayer("linear-2", Shape(hidden_size, hidden_size)));
    discriminator.addLayer(new LeakyReLUActivation("leaky-relu-2", 0.2));
    discriminator.addLayer(new LinearLayer("linear-3", Shape(hidden_size, 1)));
    discriminator.addLayer(new SigmoidActivation("sigmoid-1"));
    discriminator.setCostFunction(&bce_cost);

    NeuralNetwork generator;
    generator.addLayer(new LinearLayer("linear-1", Shape(latent_size, hidden_size)));
    generator.addLayer(new ReLUActivation("relu-1"));
    generator.addLayer(new LinearLayer("linear-2", Shape(hidden_size, hidden_size)));
    generator.addLayer(new ReLUActivation("relu-2"));
    generator.addLayer(new LinearLayer("linear-2", Shape(hidden_size, image_size)));
    generator.addLayer(new TanhActivation("tanh-1"));
    generator.setCostFunction(&bce_cost);


    ParameterGroup discriminator_params = discriminator.get_all_optimizable_parameters();
    AdamOptimizer d_optimizer(learning_rate);
    d_optimizer.set_parameter_group(discriminator_params);

    ParameterGroup generator_params = generator.get_all_optimizable_parameters();
    AdamOptimizer g_optimizer(learning_rate);
    d_optimizer.set_parameter_group(generator_params);


    Matrix outputs;
    Matrix fake_images;
    float d_loss = 0.0;
    float g_loss = 0.0;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        for (int batch = 0; batch < train_dataset.getNumOfBatches() - 1; batch++)
        {

            Matrix real_labels = Matrix(Shape(batch_size, 1));
            real_labels.allocateMemory();
            real_labels.fill_ones_on_device();
            // printMatrixCUDA(real_labels);

            Matrix fake_labels = Matrix(Shape(batch_size, 1));
            fake_labels.allocateMemory();
            fake_labels.fill_zeros_on_device();

            // ==================================
            //      Train the Discriminator
            // ==================================

            outputs = discriminator.forward(train_dataset.getBatches().at(batch));
            float d_loss_real = bce_cost.cost(outputs, real_labels);
            // float real_score = outputs;

            Matrix noise = Matrix(Shape(batch_size, latent_size));
            noise.allocateMemory();
            noise.random_init_on_device();
            fake_images = generator.forward(noise);
            outputs = discriminator.forward(fake_images);
            float d_loss_fake = bce_cost.cost(outputs, fake_labels);
            // float fake_score = outputs;

            // d_loss = d_loss_real + d_loss_fake;
            discriminator.backprop(outputs, fake_labels);
            d_optimizer.step();

            // ===============================
            //      Train the Generator
            // ===============================

            noise.random_init_on_device();
            fake_images = generator.forward(noise);
            outputs = discriminator.forward(fake_images);

            // g_loss = bce_cost.cost(outputs, real_labels);
            generator.backprop(fake_images, real_labels);
            g_optimizer.step();

        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch
                << std::endl;
        }
    }

    std::cout << "Training Process Complete. " << std::endl;
    std::cout << "Trying to generate a fake image... " << std::endl;

    Matrix noise = Matrix(Shape(batch_size, latent_size));
    noise.allocateMemory();
    noise.random_init_on_device();
    fake_images = generator.forward(noise);
    printMatrixCUDA(fake_images);
}