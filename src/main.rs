use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassEndInfo, SubpassBeginInfo, SubpassContents
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::{self, DescriptorSetLayoutCreateFlags}, CopyDescriptorSet, PersistentDescriptorSet, WriteDescriptorSet}, device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags
        }, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageUsage, ImageLayout}, instance::{Instance, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState, AttachmentBlend, BlendFactor, BlendOp},depth_stencil::{DepthState, DepthStencilState, CompareOp}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{RasterizationState, CullMode, FrontFace}, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo
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
    fs::*,
    fs,
    io::{self, BufRead},
    path::Path,
};
use nalgebra_glm::{half_pi, identity, look_at, perspective, rotate_normalized_axis, translate, vec3, vec4, TMat4, TVec3, TVec4, Vec3};
use bytemuck::{Pod, Zeroable};

fn main() {
    let mut mvp = MVP::new();
    let mut amb = AMB::new();
    let mut drb_r = DRB::new();
    let mut drb_g = DRB::new();
    let mut drb_b = DRB::new();

    mvp.view = look_at(&vec3(0.0, 0.0, 3.0), &vec3(0.0, 0.0, 0.0), &vec3(0.0, 1.0, 0.0));
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, 0.0));

    amb.color = vec3(0.0, 0.0, 0.0);
    amb.intensity = 0.1;

    drb_r.position = vec4(-4.0, 0.0, 0.0, 1.0);
    drb_r.color = vec3(1.0, 0.0, 0.0);

    drb_g.position = vec4(0.0, -6.0, 0.0, 1.0);
    drb_g.color = vec3(0.0, 1.0, 0.0);
    
    drb_b.position = vec4(4.0, 0.0, 0.0, 1.0);
    drb_b.color = vec3(0.0, 0.0, 1.0);

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
        position: Vec3,
        #[format(R32G32B32_SFLOAT)]
        normal: Vec3,
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

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Pod, Zeroable)]
    struct AMB {
        color: TVec3<f32>,
        intensity: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Pod, Zeroable)]
    struct DRB {
        position: TVec4<f32>,
        color: TVec3<f32>,
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

    impl AMB {
        fn new() -> AMB {
            AMB {
                color: vec3(0.0, 0.0, 0.0),
                intensity: 0.0,
            }
        }
    }

    impl DRB {
        fn new() -> DRB {
            DRB {
                position: vec4(0.0, 0.0, 0.0, 1.0),
                color: vec3(0.0, 0.0, 0.0),
            }
        }
    }

    let mut vertices = Vec::new();
    let mut v_positions = Vec::new();
    let mut v_normals = Vec::new();

    let lines = fs::read_to_string("./src/data/models/monkey.obj").expect("Should have been able to read the file");

    for line in lines.lines() {
        if &line[..2] == "v " {
            let numbers = &line[2..];
            let mut numbers_split = numbers.split(' ');

            v_positions.push(
                vec3(
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                    )
            );
        }else if &line[..2] == "vn" {
            let numbers = &line[3..];
            let mut numbers_split = numbers.split(' ');

            v_normals.push(
                vec3(
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                        numbers_split.next().unwrap().parse::<f32>().unwrap(),
                    )
            );
        }else if &line[..2] == "f " {
            let indice_numbers = &line[2..];
            let mut indice_numbers_split = indice_numbers.split(' ');
            
            verticeIndice(indice_numbers_split.next().unwrap(), &v_positions, &v_normals, &mut vertices);
            verticeIndice(indice_numbers_split.next().unwrap(), &v_positions, &v_normals, &mut vertices);
            verticeIndice(indice_numbers_split.next().unwrap(), &v_positions, &v_normals, &mut vertices);

            fn verticeIndice(indice_numbers_split: &str, v_positions: &Vec<Vec3>, v_normals: &Vec<Vec3>, vertices: &mut Vec<Vertex>){
                let mut indice_numbers_split_split = indice_numbers_split.split("//");
                vertices.push(
                    Vertex{
                        position: v_positions[indice_numbers_split_split.next().unwrap().parse::<usize>().unwrap() - 1],
                        normal: v_normals[indice_numbers_split_split.next().unwrap().parse::<usize>().unwrap() - 1],
                        color: [1.0, 1.0, 1.0],
                    }
                );
            }
        }
    }

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
                depth_stencil: {depth},
                input: [color, normals]
            }
        ]
    ).unwrap();

    let rotat_direction = Arc::new(AtomicBool::new(true));
    let rotat_dir = rotat_direction.clone();

    // deferred pipeline
    let deferred_pipeline = {

        let dv = deferred_vert::load(device.clone()).unwrap().entry_point("main").unwrap();
        let df = deferred_frag::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = Vertex::per_vertex().definition(&dv.info().input_interface).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(dv),
            PipelineShaderStageCreateInfo::new(df),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
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
                    deferred_pass.num_color_attachments(),
                    ColorBlendAttachmentState::default()
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(deferred_pass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        ).unwrap()
    };

    // ambient pipeline
    let ambient_pipeline = {

        let av = ambient_vert::load(device.clone()).unwrap().entry_point("main").unwrap();
        let af = ambient_frag::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = Vertex::per_vertex().definition(&av.info().input_interface).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(av),
            PipelineShaderStageCreateInfo::new(af),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
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
                    depth: Some(DepthState {
                        compare_op: CompareOp::LessOrEqual,
                        ..DepthState::simple()
                    }),
                    ..Default::default()
                }),
                color_blend_state:
                    Some(ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend{
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        color_blend_op: BlendOp::Add,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Max,
                    })),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(lighting_pass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        ).unwrap()
    };

    // directional pipeline
    let directional_pipeline = {

        let dv = directional_vert::load(device.clone()).unwrap().entry_point("main").unwrap();
        let df = directional_frag::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = Vertex::per_vertex().definition(&dv.info().input_interface).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(dv),
            PipelineShaderStageCreateInfo::new(df),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
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
                    depth: Some(DepthState {
                        compare_op: CompareOp::LessOrEqual,
                        ..DepthState::simple()
                    }),
                    ..Default::default()
                }),
                color_blend_state:
                    Some(ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend{
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        color_blend_op: BlendOp::Add,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Add,
                    })),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(lighting_pass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        ).unwrap()
    };

    // buffers
    let uniform_buffer = {
        Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }, 
            mvp,
        )
        .unwrap()
    };

    let ambient_buffer = {
        Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE |  MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            amb,
        ).unwrap()
    };

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let (mut framebuffers, mut color_buffer, mut normal_buffer) = window_size_dependent_setup(
        &images,
        render_pass.clone(),
        &mut viewport,
        &memory_allocator
    );

    let deferred_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
    let deferred_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        deferred_layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
        []
    ).unwrap();

    let ambient_layout = ambient_pipeline.layout().set_layouts().get(0).unwrap();
    let mut ambient_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        ambient_layout.clone(),
        [
            WriteDescriptorSet::image_view(0, color_buffer.clone()),
            WriteDescriptorSet::buffer(2, uniform_buffer.clone()),
            WriteDescriptorSet::buffer(3, ambient_buffer.clone()),
        ],
        []
    ).unwrap();

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

            let (new_framebuffers, new_color_buffer, new_normal_buffer) =
                window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut viewport,
                    &memory_allocator,
                );

            swapchain = new_swapchain;
            framebuffers = new_framebuffers;
            color_buffer = new_color_buffer;
            normal_buffer = new_normal_buffer;

            let ambient_layout = ambient_pipeline.layout().set_layouts().get(0).unwrap();
            ambient_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                ambient_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer.clone()),
                    WriteDescriptorSet::buffer(2, uniform_buffer.clone()),
                    WriteDescriptorSet::buffer(3, ambient_buffer.clone()),
                ],
                []
            ).unwrap();

            recr_swapch.store(false, Ordering::Relaxed);
        }

        let mock = mock.lock().unwrap();

        rotate_amount += if rotat_dir.load(Ordering::Relaxed) {0.01} else {-0.01};

        let rotata = rotate_normalized_axis(
            &mock,
            rotate_amount as f32,
            &vec3(0.0, 1.0, 0.0),
        );
        let model = rotate_normalized_axis(
            &mock,
            //rotate_amount as f32,
            3.14,
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

        let clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some(1.0.into())];
     
        let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let mut commands = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        commands
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
            .bind_pipeline_graphics(deferred_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                deferred_pipeline.layout().clone(),
                0,
                deferred_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo{
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        // directional r now i thinkies

        let directional_layout = directional_pipeline.layout().set_layouts().get(0).unwrap();

        let directional_buffer = {
            Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                drb_r,
            ).unwrap()
        };

        let mut directional_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_buffer.clone()),
                WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer.clone()),
                WriteDescriptorSet::buffer(4, directional_buffer.clone()),
            ],
            [] 
        ).unwrap();

        commands
            .bind_pipeline_graphics(directional_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        // directional g now i thinkies

        let directional_layout = directional_pipeline.layout().set_layouts().get(0).unwrap();

        let directional_buffer = {
            Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                drb_g,
            ).unwrap()
        };

        let mut directional_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_buffer.clone()),
                WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer.clone()),
                WriteDescriptorSet::buffer(4, directional_buffer.clone()),
            ],
            [] 
        ).unwrap();

        commands
            .bind_pipeline_graphics(directional_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        // directional b now i thinkies

        let directional_layout = directional_pipeline.layout().set_layouts().get(0).unwrap();

        let directional_buffer = {
            Buffer::from_data(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                drb_b,
            ).unwrap()
        };

        let mut directional_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_buffer.clone()),
                WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, uniform_buffer.clone()),
                WriteDescriptorSet::buffer(4, directional_buffer.clone()),
            ],
            [] 
        ).unwrap();

        commands
            .bind_pipeline_graphics(directional_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();


        // ambient now
        commands
            .bind_pipeline_graphics(ambient_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(
                SubpassEndInfo::default()
            )
            .unwrap();

        let command_buffer = commands.build().unwrap();

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
                *muck = translate(&identity(), &vec3(0.0, 0.0, 0.0));
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
) -> (
    Vec<Arc<Framebuffer>>,
    Arc<ImageView>,
    Arc<ImageView>,
){

    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    let color_buffer = ImageView::new_default(
        Image::new(
            allocator.clone(),
            ImageCreateInfo{
                usage: ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                format: Format::A2B10G10R10_UNORM_PACK32,
                extent: extent,
                ..Default::default()
            },
            AllocationCreateInfo::default()).unwrap(),
        ).unwrap();

    let normal_buffer = ImageView::new_default(
        Image::new(
            allocator.clone(),
            ImageCreateInfo{
                usage: ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                format: Format::R16G16B16A16_SFLOAT,
                extent: extent,
                ..Default::default()
            },
            AllocationCreateInfo::default()).unwrap(),
        ).unwrap();

    let depth_buffer = ImageView::new_default(
        Image::new(
            allocator.clone(), 
            ImageCreateInfo{
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                format: Format::D16_UNORM,
                extent: extent,
                ..Default::default()
            },
            AllocationCreateInfo::default()).unwrap(),
        ).unwrap();
    
    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view, 
                        color_buffer.clone(),
                        normal_buffer.clone(),
                        depth_buffer.clone()
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (framebuffers, color_buffer.clone(), normal_buffer.clone())
}

// loading the shaders here
mod deferred_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shaders/deferred.vert",
    }
}

mod deferred_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shaders/deferred.frag",
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
