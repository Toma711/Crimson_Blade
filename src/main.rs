use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::{self, DescriptorSetLayoutCreateFlags}, CopyDescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags
        },
    image::{view::ImageView, Image},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo
        }, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{self, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, future::FenceSignalFuture, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use nalgebra_glm::{half_pi, identity, look_at, perspective, rotate_normalized_axis, translate, vec3, TMat4, pi};
use bytemuck::{Pod, Zeroable};

fn main() {
    let mut mvp = MVP::new();

    mvp.view = look_at(&vec3(0.0, 0.0, 0.1), &vec3(0.0, 0.0, 0.0), &vec3(0.0, 1.0, 0.0));
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -1.0));

    let event_loop = winit::event_loop::EventLoopBuilder::new()
        .build()
        .expect("unable to create winit event loop");

    let instance = {
        let library = VulkanLibrary::new().unwrap();
        let extensions = Surface::required_extensions(&event_loop);

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                max_api_version: Some(Version::V1_1),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let window = Arc::new(winit::window::WindowBuilder::new()
    .build(&event_loop)
    .unwrap());

    let surface = vulkano::swapchain::Surface::from_window(
        instance.clone(),
        window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS) && 
                        p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("No suitable physical device found");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let supported_formats = device.physical_device().surface_formats(&surface, Default::default()).unwrap();
        let (chosen_format, _color_space) = supported_formats.first().unwrap();

        let usage = vulkano::image::ImageUsage::COLOR_ATTACHMENT;
        let alpha = vulkano::swapchain::CompositeAlpha::Opaque;

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        let image_extent: [u32; 2] = window.inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: 2,
                image_format: *chosen_format,
                image_extent,
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    #[repr(C)]
    #[derive(BufferContents, Vertex)]
    struct Vertex {
        #[format(R32G32B32_SFLOAT)]
        position: [f32; 3],
        #[format(R32G32B32_SFLOAT)]
        color: [f32; 3],
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Pod, Zeroable)]
    struct MVP {
        model: TMat4<f32>,
        view: TMat4<f32>,
        projection: TMat4<f32>,
    }

    impl MVP {
        fn new() -> MVP {
            MVP {
                model: identity(),
                view: identity(),
                projection: identity(),
            }
        }
    }

    let vertices = [
        Vertex {
            position: [-0.5, 0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.0, -0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: r"
                #version 450
                layout(location = 0) in vec3 position;

                layout(location = 1) in vec3 color;

                layout(location = 0) out vec3 out_color;

                layout(set = 0, binding = 0) uniform MVP_Data {
                    mat4 model;
                    mat4 view;
                    mat4 projection;
                } uniforms;

                void main() {                    
                    mat4 worldview = uniforms.view * uniforms.model;
                    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                    out_color = color;
                }
            "
        }
    }


    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450
                layout(location = 0) in vec3 color;

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(color, 1.0);
                }
            "
        }
    }

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let pipeline = {
        let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = Vertex::per_vertex().definition(&vs.info().input_interface).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(device.clone(), None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default()
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let rotation_start = Instant::now();

    let uniform_buffer = {
        Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter:: HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }, 
            mvp,
        )
        .unwrap()
    };

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
        []
    )
    .unwrap();

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let recreate_swapchain = Arc::new(AtomicBool::new(false));

    let recr_swapch = recreate_swapchain.clone(); 

    // rendering thread
    let rendering_handler = std::thread::spawn(move || loop {
        // do render operations
        // garbo collecto?

        if recr_swapch.load(Ordering::Relaxed) {
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let image_extent: [u32; 2] = window.inner_size().into();

            let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
            let projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);
            uniform_buffer.write().unwrap().projection = projection;

            let (new_swapchain, new_images) = swapchain.recreate(SwapchainCreateInfo {
                image_extent,
                ..swapchain.create_info()
            })
            .expect("failed to create swapchain");

            swapchain = new_swapchain;
            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
            recr_swapch.store(false, Ordering::Relaxed);
        }

        let elapsed = rotation_start.elapsed().as_secs() as f64
            + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
        let elapsed_as_radians = elapsed * pi::<f64>() / 180.0 * 30.0;
        let model = rotate_normalized_axis(
            &mvp.model,
            elapsed_as_radians as f32,
            &vec3(0.0, 0.0, 1.0),
        );
        uniform_buffer.write().unwrap().model = model;

        let (image_index, suboptimal, acquire_feature) =
            match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {recr_swapch.store(true, Ordering::Relaxed); return;}
                Err(e) => panic!("Faled to acquire next image: {:?}", e)
            };

        if suboptimal {
            recr_swapch.store(true, Ordering::Relaxed);
        }

        let clear_values = vec![Some([1.0, 0.80, 0.85, 1.0].into())];

        let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo{
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo{
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        cmd_buffer_builder
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        cmd_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        let command_buffer = cmd_buffer_builder.build().unwrap();

        let dabadoo = 
            acquire_feature
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

        match dabadoo {
            Ok(mut asd) => {
                asd.wait(None).unwrap();
                asd.cleanup_finished();
            },
            Err(_) => recr_swapch.store(true, Ordering::Relaxed),
        }

            // put gameloop here

    });

    // window creation thread
    event_loop.run(move |event, win_target| match event {
        Event::WindowEvent { 
            event: WindowEvent::CloseRequested,
            ..
        } => {
            win_target.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain.store(true, Ordering::Relaxed);
        }
        _ => {}
    })
    .expect("unable to run event loop");
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>>{
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}