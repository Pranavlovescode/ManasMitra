import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

export async function GET() {
  try {
    console.log('Testing MongoDB connection...');
    
    // Test MongoDB connection
    await connectDB();
    console.log('✅ MongoDB connected successfully');
    
    // Test User model
    const userCount = await User.countDocuments();
    console.log(`📊 Total users in database: ${userCount}`);
    
    // Test creating a sample user (optional)
    const testUser = {
      clerkId: 'test-user-' + Date.now(),
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User',
      role: 'patient',
      profileComplete: false,
      isActive: true,
    };
    
    const newUser = new User(testUser);
    await newUser.save();
    console.log('✅ Test user created successfully');
    
    // Clean up test user
    await User.deleteOne({ clerkId: testUser.clerkId });
    console.log('✅ Test user cleaned up');
    
    return NextResponse.json({
      success: true,
      message: 'Database connection successful',
      userCount,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Database test failed:', error);
    return NextResponse.json({
      success: false,
      error: error.message,
      stack: error.stack
    }, { status: 500 });
  }
}
