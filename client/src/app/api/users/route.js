import { auth, currentUser } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

// GET /api/users - Get current user profile
export async function GET(req) {
  try {
    console.log('🔍 [API] Getting user profile...');
    
    const { userId } = await auth(req);
    console.log('🔑 [API] Auth userId:', userId);
    
    if (!userId) {
      console.log('❌ [API] No userId found in auth context');
      return NextResponse.json({ error: 'Unauthorized - No user session found' }, { status: 401 });
    }

    await connectDB();
    
    let user = await User.findOne({ clerkId: userId });

    // If user doesn't exist in MongoDB, create them from Clerk data
    if (!user) {
      const clerkUser = await currentUser();

      if (!clerkUser) {
        return NextResponse.json(
          { error: "User not found in Clerk" },
          { status: 404 }
        );
      }

      // Create user in MongoDB with Clerk data using safe method
      user = User.createSafeUser({
        clerkId: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        firstName: clerkUser.firstName || '',
        lastName: clerkUser.lastName || '',
        role: clerkUser.publicMetadata?.role || clerkUser.unsafeMetadata?.role || 'patient',
        profileImage: clerkUser.imageUrl,
        profileComplete: false,
        isActive: true
      });

      await user.save();
      console.log('✅ User created from Clerk data:', user.email);
    }

    console.log('✅ [API] User profile fetched successfully');
    return NextResponse.json(user);
  } catch (error) {
    console.error("❌ [API] Error fetching user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// PUT /api/users - Update current user profile
export async function PUT(req) {
  try {
    console.log('🔄 [API] Updating user profile...');
    
    const { userId } = auth();
    console.log('🔑 [API] Auth userId:', userId);
    
    if (!userId) {
      console.log('❌ [API] No userId found in auth context for PUT');
      return NextResponse.json({ error: 'Unauthorized - No user session found' }, { status: 401 });
    }

    const body = await req.json();

    await connectDB();

    let user = await User.findOne({ clerkId: userId });

    // If user doesn't exist, create them first
    if (!user) {
      const clerkUser = await currentUser();

      if (!clerkUser) {
        return NextResponse.json(
          { error: "User not found in Clerk" },
          { status: 404 }
        );
      }

      user = User.createSafeUser({
        clerkId: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        firstName: clerkUser.firstName || '',
        lastName: clerkUser.lastName || '',
        role: clerkUser.publicMetadata?.role || clerkUser.unsafeMetadata?.role || 'patient',
        profileImage: clerkUser.imageUrl,
        profileComplete: false,
        isActive: true
      });
    }

    console.log('📥 User update request body:', JSON.stringify(body, null, 2));
    
    // Filter out any extra fields not in schema before updating
    const allowedFields = ['email', 'firstName', 'lastName', 'profileImage', 'profileComplete'];
    const filteredBody = {};
    for (const field of allowedFields) {
      if (body[field] !== undefined) {
        filteredBody[field] = body[field];
      }
    }
    
    console.log('🔒 Filtered update data:', JSON.stringify(filteredBody, null, 2));
    
    // Use safe update method
    User.updateSafeUser(user, filteredBody);
    
    await user.save();
    
    console.log('💾 User saved successfully. Final user object keys:', Object.keys(user.toObject()));

    return NextResponse.json(user);
  } catch (error) {
    console.error("Error updating user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// DELETE /api/users - Soft delete current user
export async function DELETE(req) {
  try {
    console.log('🗑️ [API] Deleting user profile...');
    
    const { userId } = auth();
    console.log('🔑 [API] Auth userId:', userId);
    
    if (!userId) {
      console.log('❌ [API] No userId found in auth context for DELETE');
      return NextResponse.json({ error: 'Unauthorized - No user session found' }, { status: 401 });
    }

    await connectDB();

    const user = await User.findOneAndUpdate(
      { clerkId: userId, isActive: { $ne: false } }, // Only update if not already inactive
      { isActive: false },
      { new: true }
    );

    if (!user) {
      return NextResponse.json({ error: 'User not found or already deactivated' }, { status: 404 });
    }

    console.log('🗑️ User deactivated:', user.email);
    return NextResponse.json({ 
      message: 'User deactivated successfully',
      user: User.getCleanUserData(user)
    });
  } catch (error) {
    console.error("Error deactivating user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
