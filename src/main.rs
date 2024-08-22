use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::{self, DescriptorSetLayoutCreateFlags}, CopyDescriptorSet, PersistentDescriptorSet, WriteDescriptorSet}, device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags
        }, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageUsage}, instance::{Instance, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},depth_stencil::{DepthState, DepthStencilState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{RasterizationState, CullMode, FrontFace}, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo
        }, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, swapchain::{self, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo}, sync::{self, future::FenceSignalFuture, GpuFuture}, Validated, Version, VulkanError, VulkanLibrary
};
use winit::{
    event::*,
    event_loop::{self, ControlFlow, EventLoop},
    window::*,
    keyboard::{Key, NamedKey},
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
        Mutex,
    },
    time::Instant,
};
use nalgebra_glm::{half_pi, identity, look_at, perspective, rotate_normalized_axis, translate, vec3, TMat4, pi, sqrt, cross};
use bytemuck::{Pod, Zeroable};

fn main() {
    let mut mvp = MVP::new();

    mvp.view = look_at(&vec3(0.0, 0.0, 3.0), &vec3(0.0, 0.0, 0.0), &vec3(0.0, 1.0, 0.0));
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -1.0));

    let mock = Arc::new(Mutex::new(mvp.model));
    let muck = mock.clone();

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
        normal: [f32; 3],
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

    let mut vertices = [
        // With indices
        // Vertex {
        //     position: [-1.0, 3.0_f32.sqrt()/3.0, 0.0],
        //     color: [1.0, 0.0, 0.0],
        // },
        // Vertex {
        //     position: [1.0, 3.0_f32.sqrt()/3.0, 0.0],
        //     color: [0.0, 1.0, 0.0],
        // },
        // Vertex {
        //     position: [0.0, -2.0*3.0_f32.sqrt()/3.0, 0.0],
        //     color: [0.0, 0.0, 1.0],
        // },
        // Vertex {
        //     position: [0.0, 0.0, -(21.0_f32/9.0).sqrt()],
        //     color: [1.0, 1.0, 0.0],
        // },

        // no indices :frowny_face:
        // normals get calculated lower in poggwam
        Vertex {
            position: [0.0, -2.0*3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [-1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 0.0, 0.0],
        },

        Vertex {
            position: [1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 0.0, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, -(21.0_f32/9.0).sqrt()],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [-1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },

        Vertex {
            position: [0.0, -2.0*3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, 0.0, -(21.0_f32/9.0).sqrt()],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.0, 1.0, 0.0],
        },

        Vertex {
            position: [0.0, -2.0*3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-1.0, 3.0_f32.sqrt()/3.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, -(21.0_f32/9.0).sqrt()],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 0.0],
        },

        // other twiangles
        // Vertex {
        //     position: [1.375, 0.5, -0.5],
        //     color: [1.0, 0.0, 0.0],
        // },
        // Vertex {
        //     position: [2.625, 0.5, -0.5],
        //     color: [0.0, 1.0, 0.0],
        // },
        // Vertex {
        //     position: [2.0, -0.5, -0.5],
        //     color: [0.0, 0.0, 1.0],
        // },
        // Vertex {
        //     position: [2.0, 0.25, -1.5],
        //     color: [1.0, 1.0, 0.0],
        // },
        // Vertex {
        //     position: [-2.625, 0.5, -0.5],
        //     color: [1.0, 0.0, 0.0],
        // },
        // Vertex {
        //     position: [-1.375, 0.5, -0.5],
        //     color: [0.0, 1.0, 0.0],
        // },
        // Vertex {
        //     position: [-2.0, -0.5, -0.5],
        //     color: [0.0, 0.0, 1.0],
        // },
        // Vertex {
        //     position: [-2.0, 0.25, -1.5],
        //     color: [1.0, 1.0, 0.0],
        // },
    ];

    fn cross([a1, a2, a3]: [f32; 3], [b1, b2, b3]: [f32; 3]) -> [f32; 3] {
        [a2*b3-a3*b2, a3*b1-a1*b3,a1*b2-a2*b1]
    }

    fn normalize([a1, a2, a3]: [f32; 3]) -> [f32; 3] {
        let sqr_length = a1*a1+a2*a2+a3*a3;
        let length = sqr_length.sqrt();
        [a1/length, a2/length, a3/length]
    }

    for list in vertices.chunks_mut(3) {
        let [a, b, c] = list else { unreachable!(); };
        let ba = [b.position[0] - a.position[0], b.position[1] - a.position[1], b.position[2] - a.position[2]];
        let ca = [b.position[0] - c.position[0], b.position[1] - c.position[1], b.position[2] - c.position[2]];
        let ortho = cross(ba, ca);
        let normal = normalize(ortho);
        a.normal = normal;
        b.normal = normal;
        c.normal = normal;
    }

    let indices = [
        2, 1, 0,
        1, 3, 0,
        2, 3, 1,
        2, 0, 3,

        // 6, 5, 4,
        // 5, 7, 4,
        // 6, 7, 5,
        // 6, 4, 7,

        // 10, 9, 8,
        // 9, 11, 8,
        // 10, 11, 9,
        // 10, 8, 11,
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

    let index_buffer: Subbuffer<[u16]> = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices,
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: r"
                #version 450
                layout(location = 0) in vec3 position;

                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec3 color;

                layout(location = 0) out vec3 out_color;
                layout(location = 1) out vec3 out_normal;

                layout(set = 0, binding = 0) uniform MVP_Data {
                    mat4 model;
                    mat4 view;
                    mat4 projection;
                } uniforms;

                void main() {                    
                    mat4 worldview = uniforms.view * uniforms.model;
                    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                    out_color = color;
                    out_normal = (uniforms.model * vec4(normal, 0.0)).xyz;
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
                layout(location = 1) in vec3 normal;

                
                layout(location = 0) out vec4 f_color;

                void main() {
                    vec3 light = vec3(1.0, -1.0, 0.0);
                    float light_dist = distance(light, normal);
                    f_color = vec4(1.0, 0.0, 0.0, 1.0) * light_dist / 2; // applies the lighting
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
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    /*
    let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
        attachments: {
            final_color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            color: {
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
            normals: {
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            }
        },
        passes: [
            {
                color: [color, normals],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }
        ]
    ).unwrap();
    */

    let rotat_direction = Arc::new(AtomicBool::new(true));
    let rotat_dir = rotat_direction.clone();

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

        /*
        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();
        */

        GraphicsPipeline::new(device.clone(), None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face: FrontFace::Clockwise,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
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

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport, &memory_allocator);

    let recreate_swapchain = Arc::new(AtomicBool::new(false));

    let recr_swapch = recreate_swapchain.clone(); 
    
    let mut rotate_amount = 0.0;

    // rendering thread, handles rendering handling, handles the renders, has all the rendering code
    // a monad is a monoid in the category of endofunctors
    std::thread::spawn(move || loop {
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
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport, &memory_allocator);
            recr_swapch.store(false, Ordering::Relaxed);
        }

        let mock = mock.lock().unwrap();

        rotate_amount += if rotat_dir.load(Ordering::Relaxed) {0.01} else {-0.01};

        let rotata = rotate_normalized_axis(
            //&mvp.model,
            &mock,
            //std::f32::consts::PI*2.0/3.0,
            rotate_amount as f32,
            //0.0,
            &vec3(1.0, 0.0, 0.0),
        );
        let model = rotate_normalized_axis(
            //&mvp.model,
            &mock,
            //rotate_amount as f32,
            0.0,
            &vec3(0.0, 0.0, 1.0),
        );

        drop(mock);

        uniform_buffer.write().unwrap().model = model * rotata; // apply the rotation

        let (image_index, suboptimal, acquire_feature) =
            match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {recr_swapch.store(true, Ordering::Relaxed); return;}
                Err(e) => panic!("Faled to acquire next image: {:?}", e)
            };

        if suboptimal {
            recr_swapch.store(true, Ordering::Relaxed);
        }

        let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];

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
            // .bind_index_buffer(index_buffer.clone())
            // .unwrap();

        cmd_buffer_builder
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            //.draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
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
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput{
                event: KeyEvent{
                    logical_key: key,
                    state: ElementState::Pressed,
                    ..
                },
                ..
            },
            ..
        } => match key.as_ref() {
            Key::Named(NamedKey::Space) => {
                rotat_direction.store(!rotat_direction.load(Ordering::Relaxed), Ordering::Relaxed);
            },
            Key::Named(NamedKey::Escape) => {
                win_target.exit();
            },
            Key::Character("f") => {

                let fullscreen = if window.fullscreen().is_some() {
                    None
                } else {
                    Some(Fullscreen::Borderless(None))
                };
                
                window.set_fullscreen(fullscreen);
            },
            Key::Character("p") => {
                let mut muck = muck.lock().unwrap();
                *muck = translate(&identity(), &vec3(0.0, 0.0, -0.0));
            },
            Key::Character("m") => {
                let mut muck = muck.lock().unwrap();
                *muck = translate(&identity(), &vec3(0.0, 0.0, -5.0));
            }
            _ => (),
        }
        _ => {}
    })
    .expect("unable to run event loop");
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    allocator: &Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>>{
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];
    let depth_buffer = ImageView::new_default(
        Image::new(allocator.clone(), ImageCreateInfo{
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            format: Format::D16_UNORM,
            extent: extent,
            ..Default::default()
        },
        AllocationCreateInfo::default()).unwrap(),
    )
    .unwrap();
    
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

/*
mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/directional.vert",
    }
}

mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/directional.frag",
    }
}

mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/ambient.vert",
    }
}

mod ambient_frag{
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ambient.frag",
    }
}
*/
