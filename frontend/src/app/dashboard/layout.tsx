import { getSession } from 'next-auth/react'
import { redirect } from 'next/navigation'
import React from 'react'
import { db } from '~/server/db'

const layout = async ({children}:{children: React.ReactNode}) => {
    const session = await getSession();

    if(!session?.user?.id){
        redirect("/login")
    }
    const user = await db.user.findUniqueOrThrow({
        where: {id : session.user.id},
        select: {credits:true , email:true}
    })
  return (
    <div>
      <NavHeader credits={user.credits} email={user.email} />
      <main className='container mx-auto flex-1 py-6' >{children}</main>
    </div>
  )
}

export default layout
