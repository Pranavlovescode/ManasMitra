import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

// Test webhook handler without signature verification
export async function POST(req) {
  try {
    const body = await req.json();
    console.log('Test webhook received:', JSON.stringify(body, null, 2));
    
    await connectDB();
    
    // Simulate user.created event
    const testUserData = {
      id: 'user_test_' + Date.now(),
      email_addresses: [{
        id: 'email_1',
        email_address: 'test@example.com'
      }],
      primary_email_address_id: 'email_1',
      first_name: 'John',
      last_name: 'Doe',
      unsafe_metadata: {
        role: 'patient'
      },
      image_url: 'https://example.com/avatar.jpg'
    };
    
    // Create user like the webhook would
    const newUser = new User({
      clerkId: testUserData.id,
      email: testUserData.email_addresses[0].email_address,
      firstName: testUserData.first_name,
      lastName: testUserData.last_name,
      role: testUserData.unsafe_metadata.role,
      avatar: testUserData.image_url,
      profileComplete: false,
      isActive: true,
      lastLogin: new Date(),
    });

    await newUser.save();
    console.log('✅ Test user created successfully:', newUser.email);
    
    return NextResponse.json({
      success: true,
      message: 'Test user created successfully',
      user: {
        clerkId: newUser.clerkId,
        email: newUser.email,
        role: newUser.role,
        profileComplete: newUser.profileComplete
      }
    });
    
  } catch (error) {
    console.error('❌ Test webhook failed:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}